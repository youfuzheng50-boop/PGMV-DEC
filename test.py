import os

import numpy as np
import torch.nn.functional as F
import pandas as pd
import torch
from collections import defaultdict, Counter
from sklearn.metrics import confusion_matrix, classification_report
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import seaborn as sns
from models.MLP import FiveClassClassifier, FiveClassClassifier_tcga,MLPClassifier


def load_data(csv_file, feature_folder):
    """加载 CSV 并将数据转换为字典格式."""
    data = pd.read_csv(csv_file)
    wsi_features = defaultdict(list)
    labels = {}

    # 定义标签映射
    label_mapping = {'J': 1, 'N': 2, 'T': 3, 'Z': 4}
    # label_mapping = {'HGSC': 1, 'LGSC': 2, 'CC': 3, 'MC': 4, 'EC':5}
    for index, row in data.iterrows():
        slide_id = row['slide_id']
        label = row['label']

        # 将标签转换为数字
        numeric_label = label_mapping.get(label)

        # 构造特征文件的完整路径
        feature_path = os.path.join(feature_folder, f"{slide_id}.pt")

        # 加载对应的 patch 特征向量
        if os.path.exists(feature_path):
            feature = torch.load(feature_path)
            wsi_features[slide_id].append(feature)
            labels[slide_id] = numeric_label
        else:
            print(f"特征文件 {feature_path} 不存在，跳过该条目。")

    return wsi_features, labels


def classify_and_aggregate(model, wsi_features, device, num_classes, batch_size=4096):
    """
    对WSI进行分类和聚合。

    1. 硬预测 (Acc/F1): 通过多数投票得到。
    2. 软概率 (AUC): 通过平均非背景 Patch 的 Logits，再应用 Softmax 得到。
    """
    model.eval()
    wsi_hard_predictions = {}  # 存储多数投票结果 (用于 Acc/F1/CM)
    wsi_soft_probabilities = {}  # 存储 Logits 平均后 Softmax 的概率 (用于 AUC)

    with torch.no_grad():
        for wsi_id, features in tqdm(wsi_features.items(), desc='Processing WSIs'):

            all_patch_preds_list = []  # 存储所有 patch 的硬预测 (用于投票)
            all_patch_logits_list = []  # 存储所有 patch 的 Logits (用于平均)
            features = features[0]
            # --- 阶段一：批量处理 Patch 并收集数据 ---
            for i in range(0, len(features), batch_size):
                batch_features = features[i:i + batch_size].to(device)

                # 模型输出 Logits (形状: [batch_size, num_classes])
                outputs = model(batch_features)

                # 1. 收集 Logits：用于 WSI 概率的平均池化
                all_patch_logits_list.append(outputs.cpu())

                # 2. 计算 Patch 标签：用于 WSI 多数投票
                batch_preds = torch.argmax(outputs, dim=1)
                all_patch_preds_list.extend(batch_preds.cpu().tolist())

            # --- 阶段二：WSI 级别聚合 ---

            # 过滤掉背景类别 (class 0) - 与你的原始逻辑保持一致
            non_background_indices = [
                idx for idx, pred in enumerate(all_patch_preds_list) if pred != 0
            ]

            non_background_preds = [
                pred for pred in all_patch_preds_list if pred != 0
            ]

            # 1. 计算 WSI 的硬预测 (多数投票)
            if non_background_preds:
                prediction_counts = Counter(non_background_preds)
                final_hard_prediction = prediction_counts.most_common(1)[0][0]
            else:
                final_hard_prediction = 0

            wsi_hard_predictions[wsi_id] = final_hard_prediction

            # 2. 计算 WSI 的软概率 (Logits 平均后 Softmax)
            if not all_patch_logits_list:
                # WSI 为空，使用均匀概率
                wsi_avg_prob_np = np.ones(num_classes) / num_classes
            else:
                # 合并所有 batch 的 Logits tensor
                all_patch_logits_tensor = torch.cat(all_patch_logits_list, dim=0)

                if non_background_indices:
                    # 关键：只对非背景 Patch 的 Logits 进行平均
                    non_bg_patch_logits = all_patch_logits_tensor[non_background_indices]
                    wsi_avg_logits = torch.mean(non_bg_patch_logits, dim=0)  # [num_classes]
                else:
                    # 如果全部是背景 (非0的硬标签为空)，则平均所有 Patch 的 Logits
                    wsi_avg_logits = torch.mean(all_patch_logits_tensor, dim=0)

                    # 对平均 Logits 应用 Softmax 得到概率
                wsi_avg_prob = F.softmax(wsi_avg_logits.unsqueeze(0), dim=1).squeeze(0)
                wsi_avg_prob_np = wsi_avg_prob.numpy()

            wsi_soft_probabilities[wsi_id] = wsi_avg_prob_np

    return wsi_hard_predictions, wsi_soft_probabilities


def classify_and_vote(model, wsi_features, device, num_classes, batch_size=32):
    model.eval()
    wsi_final_predictions = {}
    wsi_final_probabilities = {}
    with torch.no_grad():
        for wsi_id, features in tqdm(wsi_features.items(), desc='Processing WSIs'):
            predictions = []
            for i in range(0, len(features), batch_size):
                batch_features = torch.stack(features[i:i + batch_size]).to(device)
                outputs = model(batch_features)
                outputs = outputs.squeeze(0)
                predicted = torch.argmax(outputs, dim=1)
                predictions.extend(predicted.cpu().tolist())

            predictions = [pred for pred in predictions if pred != 0]

            if predictions:
                prediction_counts = Counter(predictions)
                final_prediction = prediction_counts.most_common(1)[0][0]
            else:
                final_prediction = 0
            # 初始化每个类别计数为0
            counts = torch.zeros(num_classes, dtype=torch.float)

            # 更新各类别计数
            for cls, count in prediction_counts.items():
                counts[cls] = count

            probabilities = F.softmax(counts, dim=0)

            wsi_final_predictions[wsi_id] = final_prediction
            wsi_final_probabilities[wsi_id] = probabilities.numpy()

    return wsi_final_predictions, wsi_final_probabilities




def calculate_metrics(true_labels, predicted_labels):
    print(classification_report(true_labels, predicted_labels,digits=4))
    overall_accuracy = accuracy_score(true_labels, predicted_labels)
    print(f"总体准确率: {overall_accuracy}")
    f1 = f1_score(true_labels, predicted_labels, average='weighted')
    print(f"F1 分数: {f1}")

    # 计算混淆矩阵
    cm = confusion_matrix(true_labels, predicted_labels)
    print("混淆矩阵:")
    print(cm)
    plot_confusion_matrix(cm, class_names=['1', '2', '3', '4'])




def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Confusion Matrix')
    plt.savefig('/home/idao/Zyf/models/oc_subtype_classficter/Autoencoder/fold1/消融')
    plt.show()

if __name__ == "__main__":
    csv_file = "/home/idao/Zyf/models/oc_subtype_classficter/Dataset/flod_1/test_data.csv"
    pt_file = '/home/idao/Zyf/data/oc_features/FEATURES_DIRECTORY/pt_files'
    # 加载数据
    wsi_features, labels = load_data(csv_file,pt_file)
    num_classes = 5
    # 进行分类和多数投票
    model = FiveClassClassifier(input_dim=1024, num_classes=5)
    # model = MLPClassifier(input_size=1024,hidden_size=128,num_classes=5)
    model.load_state_dict(torch.load("/home/idao/Zyf/models/oc_subtype_classficter/multi_view/fiveClassifier0.9433.pth"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # hard_results: 多数投票的硬标签 (用于 Acc, F1, CM)
    # soft_probabilities: Logits 平均后的软概率 (用于 AUC)
    hard_results, soft_probabilities = classify_and_aggregate(model, wsi_features, device, num_classes)

    # --- 1. 准备用于 Acc/F1/CM 的数据 (使用硬标签) ---
    # 确保 true_labels 和 predicted_labels 顺序一致
    true_labels_list = [label for _, label in labels.items()]
    predicted_labels_list = [prediction for _, prediction in hard_results.items()]

    # 计算 Acc/F1/CM
    calculate_metrics(true_labels_list, predicted_labels_list)

    # --- 2. 准备用于 AUC 的数据 (使用软概率) ---
    # 确保 true_labels 和 probabilities 的顺序一致
    slide_ids_ordered = list(labels.keys())

    # 真实的标签 (y_true)，形状: (n_samples,)，值是 1, 2, 3, 4
    true_labels_array = np.array([labels[slide_id] for slide_id in slide_ids_ordered])

    # 预测的概率 (y_pred_proba)，形状: (n_samples, 5)，包含类别 0 到 4 的概率
    all_probabilities_for_auc = np.array([soft_probabilities[slide_id] for slide_id in slide_ids_ordered])

    # 1. 💥 关键修改：切片概率数组，移除第 0 列 (背景类)
    # y_score_for_auc 的形状是 (n_samples, 4)，对应于标签 1, 2, 3, 4
    y_score_raw = all_probabilities_for_auc[:, 1:]

    # 2. 💥 关键修改：重新规范化 (Normalization)
    # 计算每行（即每个 WSI）的非背景概率和
    row_sums = y_score_raw.sum(axis=1, keepdims=True)

    # 避免除以零 (如果某个 WSI 的所有非背景概率都为 0，则设为均匀分布或保留原样)
    # 这里假设所有 WSI 至少有一个非零的概率和
    row_sums[row_sums == 0] = 1.0  # 如果和为0，设为1，避免NaN，让其保留均匀分布

    # 规范化：将每行的概率除以该行的和
    y_score_for_auc = y_score_raw / row_sums

    # 验证：检查 y_score_for_auc.sum(axis=1) 是否接近 1.0 (可选)
    # print("规范化后每行之和（应接近 1.0）:", y_score_for_auc.sum(axis=1))

    # 定义我们关心的类别 (即 label_mapping 中的非背景类别)
    class_labels_of_interest = np.unique(true_labels_array)
    class_labels_of_interest = class_labels_of_interest[class_labels_of_interest != 0]

    print("\n--- 计算宏平均 AUC (OvR) ---")
    try:
        # 现在 y_score_for_auc 是规范化的 4 列概率，与 4 个标签匹配
        auc_score_macro = roc_auc_score(
            true_labels_array,  # 原始整数标签 [1, 2, 4, 3, ...]
            y_score_for_auc,  # 规范化的 4 列概率 (P'(1) 到 P'(4))
            multi_class='ovr',  # One-vs-Rest 策略
            average='macro',  # 计算宏平均
            # 移除 labels 参数，因为列数和标签数已匹配
        )
        print(f"宏平均 AUC (Macro-Average AUC, OvR): {auc_score_macro:.4f}")

    except ValueError as e:
        print(f"计算 AUC 时出错: {e}")
        print("提示: 确保每个被评估的类别 (1, 2, 3, 4) 在测试集中至少包含一个正样本和一个负样本。")




# 引入绘制 ROC 曲线所需的工具
    from sklearn.preprocessing import LabelBinarizer  # 用于 One-Hot 编码真实标签
    from sklearn.metrics import roc_curve, auc
    import matplotlib.pyplot as plt

    # 类别标签名称（用于图例，基于你的 label_mapping）
    # 确保 class_names 只包含你要评估的类别 (1, 2, 3, 4, 5)
    # 你的 label_mapping 是 {'HGSC': 1, 'LGSC': 2, 'CC': 3, 'MC': 4, 'EC':5 }
    # 假设类别 0 是背景，不需要绘制。
    # 检查 true_labels_array 中的最大值，确保它在 class_names 范围内。

    # 根据你的 num_classes=6 和标签 1-5，我们关注 5 个类别。
    class_names = {1: 'J',  2: 'N', 3: 'T', 4: 'Z'}

    # --- 1. 准备 OvR 绘图数据 ---
    # a. 将真实的整数标签转换为 One-Hot 编码
    lb = LabelBinarizer()
    # 适应于所有可能的标签 (1, 2, 3, 4)
    lb.fit(np.array(list(class_names.keys())))
    binarized_true_labels = lb.transform(true_labels_array)  # 形状: (n_samples, 4)

    # b. 预测概率 y_score_for_auc 形状为 (n_samples, 5)，对应标签 1 到 4
    # 注意：这里假设你的 num_classes=6 意味着类别是 0 到 5，
    # 你的 y_score_for_auc 应该是 5 列 (1:5)。我们沿用之前的切片逻辑。
    # **重要：重新确认 y_score_for_auc 的形状：它应该有 5 列 (对应 1, 2, 3, 4, 5)**
    # 你的 label_mapping 有 5 个类别 (1-5)，num_classes=6 (0-5)。
    # 因此 y_score_raw = all_probabilities_for_auc[:, 1:] 应该是 5 列。

    # 为每个类别计算 ROC 曲线数据
    fpr = dict()
    tpr = dict()
    # roc_auc = dict() # 可以在这里计算，但你已经计算了宏平均

    # 迭代类别 1 到 5
    evaluated_class_ids = list(class_names.keys())

    # c. 修正 y_score_for_auc 的列数以匹配类别数量 (5)
    # y_score_for_auc 是规范化后的 P(1)到P(5)，共 5 列。
    # binarized_true_labels 也是 5 列，对应类别 1到5。

    print("\n--- 绘制 ROC 曲线 (OvR) ---")

    plt.figure(figsize=(10, 8))

    for i, class_id in enumerate(evaluated_class_ids):
        # binarized_true_labels[:, i] 是类别 class_id 的二元真实标签
        # y_score_for_auc[:, i] 是类别 class_id 的预测概率

        # 计算当前类别的 FPR, TPR
        fpr[class_id], tpr[class_id], _ = roc_curve(
            binarized_true_labels[:, i],
            y_score_for_auc[:, i]  # 使用规范化后的概率
        )

        # 计算当前类别的 AUC (与宏平均 AUC 内部的单类别 AUC 相同)
        roc_auc_single = auc(fpr[class_id], tpr[class_id])

        # 绘制曲线
        plt.plot(fpr[class_id], tpr[class_id],
                 label=f'{class_names[class_id]} (AUC = {roc_auc_single:.4f})')

    # 绘制对角线 (随机分类器)
    plt.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.50)')

    # 添加宏平均 AUC 文本（如果计算成功）
    try:
        plt.text(0.8, 0.2, f'Macro-AUC: {auc_score_macro:.4f}',
                 fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
    except NameError:
        pass  # 如果auc_score_macro没有计算成功，则跳过

    # 设置图表标题和轴标签
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('Receiver Operating Characteristic (ROC) Curves (One-vs-Rest)')
    plt.legend(loc='lower right')
    plt.grid(True)

    # 保存图像
    roc_save_path = '/home/idao/Zyf/models/oc_subtype_classficter/multi_view/roc_curves.png'
    plt.savefig(roc_save_path)
    print(f"ROC 曲线已保存到: {roc_save_path}")

    plt.show()