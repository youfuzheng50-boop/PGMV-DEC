from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

import os
import gc
import time
import pandas as pd
from tqdm import tqdm
from collections import Counter

# --- 移除的导入 ---
# 移除了大量未使用的库：
# PCA, multiprocessing, DataLoader, TensorDataset,
# BinaryClassifierMLP, FiveClassClassifier_tcga, FiveClassClassifier,
# Build_NDI, create_combined_dataset, classify_wsi_and_vote_verbose,
# ClusterSelector, SummaryWriter, ThreadPoolExecutor, as_completed, traceback,
# EmptyDataset, heapq (用 sorted 替代)

# --- 模型定义 ---
# (假设 MultiViewAutoencoder 位于此处或已正确导入)
# from models.model_autoencoder import MultiViewAutoencoder, add_noise, SharedSpecificAutoencoder
from models.model_autoencoder import MultiViewAutoencoder
from models.kmeans import update_cluster_centers


# --- 损失函数 ---
# (soft_cluster_distribution_student, calculate_target_distribution,
#  kl_loss, proto_compactness_loss, multi_view_loss 保持不变)
# ... [此处省略了您提供的所有损失函数，它们无需更改] ...
def soft_cluster_distribution_student(shared, proto_vectors, nu=0.1, temp=0.05, epsilon=1e-8):
    shared_norm = F.normalize(shared, p=2, dim=1)
    proto_norm = F.normalize(proto_vectors, p=2, dim=1)
    cos_sim = torch.mm(shared_norm, proto_norm.t())
    shifted_sim = (cos_sim + 1) / 2
    student_scores = (1 + (shifted_sim.pow(2)) / nu).pow(-(nu + 1) / 2)
    ranks = torch.argsort(torch.argsort(student_scores, dim=1), dim=1).float()
    rank_weights = 1.0 + (ranks / student_scores.size(1))
    boosted_scores = student_scores * rank_weights
    logits = boosted_scores / temp
    logits = logits - logits.max(dim=1, keepdim=True)[0]
    q = F.softmax(logits, dim=1)
    return q


def calculate_target_distribution(q, normalize_frequencies=True, epsilon=1e-8):
    f = torch.sum(q, dim=0) / q.size(0)
    q_squared = q ** 2
    p = q_squared / (f + epsilon)
    p = p / p.sum(dim=1, keepdim=True)
    return p


def kl_loss(shared, proto_vectors):
    q = soft_cluster_distribution_student(shared, proto_vectors)
    p = calculate_target_distribution(q)
    cluster_loss = F.kl_div(q.log(), p, reduction='batchmean')
    return 3 * cluster_loss


def proto_compactness_loss(proto_vectors):
    n = proto_vectors.size(0) // 2
    proto1 = proto_vectors[:n]
    proto2 = proto_vectors[n:]
    center1 = proto1.mean(dim=0, keepdim=True)
    center2 = proto2.mean(dim=0, keepdim=True)

    def cosine_similarity(a, b):
        return F.cosine_similarity(a, b, dim=-1)

    sim1 = cosine_similarity(proto1, center1)
    sim2 = cosine_similarity(proto2, center2)
    compact_loss = torch.mean(1 - sim1) + torch.mean(1 - sim2)
    separation_loss = 1 - cosine_similarity(center1, center2)
    total_loss = 5 * compact_loss + 1 * separation_loss
    return total_loss


def multi_view_loss(encode_list, p_encode_list):
    # --- 添加这个检查 ---
    # 如果列表长度小于2 (即单视图模式)，则没有多视图损失，返回 0.0
    if len(encode_list) < 2:
        # 确保返回的是一个 tensor，以便与其他损失相加
        # (假设 encode_list[0] 存在，并且在正确的设备上)
        return torch.tensor(0.0, device=encode_list[0].device, dtype=encode_list[0].dtype)
    # --- 检查结束 ---

    # 原始代码（现在是安全的，因为我们已经确保了有两个视图）
    encode_list_0 = encode_list[0]
    p_encode_list_0 = p_encode_list[0]
    encode_list_1 = encode_list[1]
    p_encode_list_1 = p_encode_list[1]

    q_12 = soft_cluster_distribution_student(encode_list_0, p_encode_list_0)
    p_12 = soft_cluster_distribution_student(encode_list_1, p_encode_list_1)
    q_21 = soft_cluster_distribution_student(encode_list_1, p_encode_list_1)
    p_21 = soft_cluster_distribution_student(encode_list_0, p_encode_list_0)

    # 多视图对齐
    loss_12 = F.kl_div(q_12.log(), p_12, reduction='batchmean')
    loss21 = F.kl_div(q_21.log(), p_21, reduction='batchmean')
    all_loss = 0.5 * loss_12 + 0.2 * loss21

    return all_loss


# --- 移除的死代码 ---
# 移除了 load_wsi_features 和 train_five_class_model，因为它们未被调用。

# --- 训练函数 (微调) ---
def train_shared_specific_model(features_list, proto_vectors_list, model, optimizer, scheduler, epochs=10,
                                update_interval=1,
                                device='cuda'):
    lambda_recon = 1.0
    criterion = nn.MSELoss()
    K = 2
    features_list = [feature.to(device) for feature in features_list]

    # K-means 初始化 (假设 update_cluster_centers 已导入)
    _, _, _, _, _ = update_cluster_centers(features_list, K, proto_vectors_list, 1)

    with torch.no_grad():
        encode_list = model.encode(*features_list)
        p_encode_list = model.encode(*proto_vectors_list)
        cluster_labels, cluster_centers, sim, cancer_label, normal_label = update_cluster_centers(encode_list, K,
                                                                                                  p_encode_list, 1)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        reconstructed_views = model(*features_list)
        encode_list = model.encode(*features_list)
        p_encode_list = model.encode(*proto_vectors_list)

        recon_loss = sum([criterion(recon, feature) for recon, feature in zip(reconstructed_views, features_list)])
        proto_loss_kl = sum([kl_loss(encode, p_encode) for encode, p_encode in zip(encode_list, p_encode_list)])
        proto_loss = sum([proto_compactness_loss(p_encode) for p_encode in p_encode_list])
        mv_loss = multi_view_loss(encode_list, p_encode_list)

        proto_weight = 1.0 if epoch > 20 else (epoch + 1) / 20.0

        loss = (lambda_recon * recon_loss + 2 * proto_loss_kl +
                proto_weight * (proto_loss + 0.1 * mv_loss))

        loss.backward()
        optimizer.step()
        scheduler.step()

        if (epoch + 1) % update_interval == 0:
            with torch.no_grad():
                # 1. 不再解包，直接将 encode 的结果赋值给 encode_list
                encode_list = model.encode(*features_list)
                p_encode_list = model.encode(*proto_vectors_list)

                # 2. 使用刚刚在 with torch.no_grad() 块中
                #    重新计算的 encode_list 和 p_encode_list 来更新聚类中心
                cluster_labels, cluster_centers, sim, cancer_label, normal_label = update_cluster_centers(encode_list,
                                                                                                          K,
                                                                                                          p_encode_list,
                                                                                                          1)

        # 优化：仅在最后几个 epoch 或特定间隔打印日志，避免刷屏
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            print(f"  Epoch [{epoch + 1}/{epochs}], Recon Loss: {recon_loss.item():.4f}, "
                  f"MV Loss: {mv_loss.item():.4f}, "
                  f"Proto KL: {proto_loss_kl.item():.4f}, Proto Compact: {proto_loss.item():.4f}, "
                  f"Total: {loss.item():.4f}")

    return cluster_labels, cluster_centers, sim, cancer_label, normal_label


# --- 移除的函数 ---
# 移除了 selected_samles 函数，其逻辑被合并到 process_bag 中，以避免全局变量。

# --- 数据加载函数 (保持不变) ---
def load_slide_features(slide_id, pt_files_paths):
    features_list = []
    for path in pt_files_paths:
        pt_file_path = os.path.join(path, slide_id + ".pt")
        if os.path.exists(pt_file_path):
            try:
                # 假设 .pt 文件是使用 torch.save(tensor, path) 保存的
                features = torch.load(pt_file_path, weights_only=False)
                features_list.append(features)
            except Exception as e:
                print(f"错误：无法加载文件 {pt_file_path}. 错误: {e}")
                return None
        else:
            print(f"文件 {pt_file_path} 未找到！")
            return None
    return slide_id, features_list


# --- 核心处理函数 (重构) ---
def process_bag(slide_id, bag_features_list, p_feature_list, bag_label, model_config, train_config, device):
    """
    重构的函数，用于处理单个 bag (WSI)：创建模型、训练、处理结果，并返回样本/标签。

    Args:
        slide_id (str): WSI ID.
        bag_features_list (list[Tensor]): 当前 WSI 的多视图特征列表。
        p_feature_list (list[Tensor]): 原型向量列表。
        bag_label (int): WSI 的真实标签。
        model_config (dict): 模型配置参数。
        train_config (dict): 训练配置参数。
        device (torch.device): 训练设备。

    Returns:
        tuple: (list[Tensor], list[int]) 选定的样本特征和对应的标签。
    """

    # 1. 初始化此 bag 专用的模型和优化器
    model = MultiViewAutoencoder(**model_config).to(device)
    optimizer = optim.Adam(model.parameters(), lr=train_config['lr'])
    scheduler = StepLR(optimizer, **train_config['scheduler_step'])

    print(f"\n开始处理 {slide_id} (标签: {bag_label})...")

    # 2. 训练模型（逐包训练）
    # 注意：返回值 cancer_label_list 和 normal_label_list 预计为 [int] 列表
    cluster_labels, _, sim, cancer_label_list, normal_label_list = train_shared_specific_model(
        bag_features_list, p_feature_list, model, optimizer, scheduler,
        epochs=train_config['epochs'], device=device
    )

    # 3. 提取聚类标签为整数 (int)
    # 修复 TypeError: unhashable type: 'list' 错误
    cancer_label_int = cancer_label_list[0]
    normal_label_int = normal_label_list[0]

    # 4. 保存相似度分数 (使用第一个视图的结果)
    sim_save_path = os.path.join(train_config['sim_save_path'], f"sim_{slide_id}.npy")
    # 注意：假设 sim[0] 是第一个视图的相似度矩阵
    np.save(sim_save_path, sim[0].cpu().numpy())

    # 5. 准备特征和聚类结果 (使用第一个视图的结果)
    # 假设 features[0] 是第一个视图的特征
    features = bag_features_list[0].cpu()
    cluster_labels = cluster_labels[0].cpu().numpy()
    similarities = sim[0].cpu().numpy()

    # 6. 打印聚类统计
    label_counts = Counter(cluster_labels)
    print(f"  聚类结果 (癌症={cancer_label_int}, 正常={normal_label_int}): {label_counts}")

    bag_selected_samples = []
    bag_selected_labels = []

    # 7. 提取样本数据

    # 辅助函数：根据标签提取 (特征, 相似度) 数据
    def extract_data(target_labels, target_sim_idx):
        data = []
        for i, label in enumerate(cluster_labels):
            if label in target_labels:
                data.append((features[i], similarities[i][target_sim_idx]))
        return data

    # a. 癌症样本
    cancer_data = extract_data({cancer_label_int}, cancer_label_int)
    cancer_data.sort(key=lambda x: x[1], reverse=True)  # 按相似度排序

    bag_selected_samples.extend([item[0] for item in cancer_data])
    bag_selected_labels.extend([bag_label] * len(cancer_data))

    # b. 正常样本 (Normal/Benign)
    normal_data = extract_data({normal_label_int}, normal_label_int)
    normal_data.sort(key=lambda x: x[1], reverse=True)

    bag_selected_samples.extend([item[0] for item in normal_data])
    bag_selected_labels.extend([0] * len(normal_data))  # 标签为 0

    # c. 其他样本 (既不是癌症也不是正常)
    all_cluster_labels = set(label_counts.keys())
    other_labels = all_cluster_labels - {cancer_label_int, normal_label_int}

    if other_labels:
        other_data = []
        for i, label in enumerate(cluster_labels):
            if label in other_labels:
                # 附上其对自己簇的相似度
                other_data.append((features[i], similarities[i][label]))

        other_data.sort(key=lambda x: x[1], reverse=True)
        bag_selected_samples.extend([item[0] for item in other_data])
        bag_selected_labels.extend([0] * len(other_data))  # 标签为 0

    print(f"  处理完毕. 提取到 {len(bag_selected_samples)} 个样本.")
    return bag_selected_samples, bag_selected_labels


# --- 主函数 (重构) ---
def main(config):
    """
    主执行函数，包含配置加载和主循环。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- 使用设备: {device} ---")

    # 1. 加载 WSI 列表
    labels_df = pd.read_csv(config['csv_file'])
    print(f"从 {config['csv_file']} 加载了 {len(labels_df)} 个 slides.")

    # 2. 根据 model_type 动态设置路径和模型参数
    model_config = config['model_config'].copy()

    if config['model_type'] == 'multi_view':
        print("--- 运行模式: Multi-View ---")
        proto_paths = [config['paths']['v1_proto'], config['paths']['v2_proto']]
        feature_paths = [config['paths']['v1_features'], config['paths']['v2_features']]
        model_config['input_dims'] = [1024, 1024]

    elif config['model_type'] == 'single_view':
        print("--- 运行模式: Single-View ---")
        proto_paths = [config['paths']['v1_proto']]
        feature_paths = [config['paths']['v1_features']]
        model_config['input_dims'] = [1024]

    else:
        raise ValueError(f"未知的 model_type: {config['model_type']}")

    # 3. 加载原型向量
    print("加载原型向量...")
    p_feature_list = []
    for p_path in proto_paths:
        try:
            # 假设原型文件是包含 'features' 键的字典
            p_data = torch.load(p_path, weights_only=False)
            p_feature_list.append(torch.tensor(p_data['features']).to(device))
        except Exception as e:
            print(f"严重错误: 无法加载原型文件 {p_path}. 错误: {e}")
            return

    # 4. 初始化结果列表
    all_selected_samples = []
    all_selected_labels = []

    start_time = time.perf_counter()

    # 5. 主循环
    for _, row in tqdm(labels_df.iterrows(), total=len(labels_df), desc="处理 Slides"):
        slide_id = row['slide_id']
        bag_label = config['label_mapping'][row['label']]

        # a. 加载当前 slide 的特征
        slide_data = load_slide_features(slide_id, feature_paths)
        if slide_data is None:
            print(f"跳过 {slide_id}: 特征加载失败.")
            continue

        _, features_list = slide_data

        # b. 处理 bag (训练、聚类、采样)
        bag_samples, bag_labels = process_bag(
            slide_id, features_list, p_feature_list, bag_label,
            model_config, config['train_config'], device
        )

        # c. 收集结果
        all_selected_samples.extend(bag_samples)
        all_selected_labels.extend(bag_labels)

        # d. 显存管理
        del slide_data, features_list, bag_samples, bag_labels
        torch.cuda.empty_cache()
        gc.collect()

    end_time = time.perf_counter()
    print(f"\n--- 任务完成 ---")
    print(f"总耗时: {end_time - start_time:.2f} 秒")
    print(f"总共收集到 {len(all_selected_samples)} 个样本。")

    # 6. 保存最终结果 (示例)
    # 您的原始代码没有保存 all_selected_samples，您可以在这里添加保存逻辑
    output_dir = config['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    torch.save(
        {"features": torch.stack(all_selected_samples), "labels": torch.tensor(all_selected_labels)},
        os.path.join(output_dir, "pseudo_labeled_dataset.pt")
    )
    print(f"伪标签数据集已保存。")


if __name__ == "__main__":
    # --- 优化：将所有配置集中在此处 ---
    CONFIG = {
        'model_type': 'single_view',  # 'multi_view' 或 'single_view'

        'csv_file': 'heatmaps/process_lists/heatmap_demo_dataset.csv',
        'label_mapping': {'J': 1, 'N': 2, 'T': 3, 'Z': 4},
        'paths': {
            'v1_proto': "/home/idao/Zyf/models/oc_subtype_classficter/Autoencoder/prototype/liang_e/new_img/new_img_features.pt",
            'v2_proto': "/home/idao/Zyf/models/oc_subtype_classficter/Autoencoder/prototype/liang_e/new_img/frequency/frequency_features.pt",
            'v1_features': '/home/idao/Zyf/data/oc_features/FEATURES_DIRECTORY/pt_files',
            'v2_features': '/home/idao/Zyf/data/oc_features/frequency/pt_files',

            # 'v3_features': '/home/idao/Zyf/data/oc_features/sharpen/pt_files'
        },

        # 'csv_file': 'heatmaps/process_lists/heatmap_demo_dataset.csv',
        # 'label_mapping': {'HGSC': 1, 'LGSC': 2, 'CC': 3, 'MC': 4, 'EC':5 },
        # 'paths': {
        #     'v1_proto': "/home/idao/Zyf/models/oc_subtype_classficter/Autoencoder/prototype/liang_e/new_img/new_img_features.pt",
        #     'v2_proto': "/home/idao/Zyf/models/oc_subtype_classficter/Autoencoder/prototype/liang_e/new_img/frequency/frequency_features.pt",
        #     'v1_features': '/home/idao/Zyf/data/UBC-OCEAN/UBC_feature/FEATURES_DIRECTORY/pt_files',
        #     'v2_features': '/home/idao/Zyf/data/oc_features/frequency/pt_files',
        # },

        'model_config': {
            'hidden_dim': 256,
            'latent_dim': 32,
            # 'input_dims' 将由 'model_type' 动态设置
        },

        'train_config': {
            'epochs': 30,
            'lr': 0.001,
            'scheduler_step': {'step_size': 20, 'gamma': 0.1},
            'sim_save_path': '/home/idao/Zyf/models/PGMV-DEC/heatmaps/sim/'
        },

        'output_dir': './results/'  # 示例：用于保存最终数据集
    }

    # 确保保存 sim 分数的目录存在
    os.makedirs(CONFIG['train_config']['sim_save_path'], exist_ok=True)

    main(CONFIG)