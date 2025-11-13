import os
import torch.nn.functional as F
import pandas as pd
import torch
from collections import defaultdict, Counter
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from tqdm import tqdm



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
    print(classification_report(true_labels, predicted_labels))
    overall_accuracy = accuracy_score(true_labels, predicted_labels)
    print(f"总体准确率: {overall_accuracy}")
    f1 = f1_score(true_labels, predicted_labels, average='weighted')
    print(f"F1 分数: {f1}")

def val_load_data(csv_file, feature_folder):
    """加载 CSV 并将数据转换为字典格式."""
    data = pd.read_csv(csv_file)
    wsi_features = defaultdict(list)
    labels = {}

    # 定义标签映射
    label_mapping = {'J': 1, 'N': 2, 'T': 3, 'Z': 4}

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