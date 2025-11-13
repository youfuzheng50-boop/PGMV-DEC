import numpy as np

from models.MLP import classify_intermediate_samples
from collections import Counter
import torch


def create_combined_dataset(cancer_samples_list, intermediate_samples_list, model, device, merge_samples=True):
    combined_samples = []
    combined_labels = []
    confidence_threshold = 0.97  # 设定的置信度阈值

    for idx, (cancer_samples, intermediate_samples) in enumerate(zip(cancer_samples_list, intermediate_samples_list)):
        if merge_samples:
            if intermediate_samples.any():  # 确保中间样本存在
                pseudo_labels, confidences = classify_intermediate_samples(intermediate_samples, model, device,
                                                                           head_idx=idx)

                # 遍历每个中间样本的伪标签及其置信度
                for i, (p, conf) in enumerate(zip(pseudo_labels, confidences)):
                    if p == 1 and conf >= confidence_threshold:
                        # 将符合条件的样本添加到组合样本中
                        combined_samples.append(intermediate_samples[i])
                        combined_labels.append(idx + 1)
            else:
                print(f"No intermediate samples for class index {idx}. Skipping classification step.")

        # 无论是否合并，都需要将 cancer_samples 添加到结果中
        if cancer_samples.any():  # 检查是否有癌症样本
            combined_samples.extend(cancer_samples)
            combined_labels.extend([idx + 1] * len(cancer_samples))
        else:
            print(f"No cancer samples for class index {idx}. Skipping this index.")

        print(
            f"Intermediates: {len(intermediate_samples)}, Pseudo-labels: {len(pseudo_labels) if merge_samples else 'N/A'}")

    # 将列表转换为 numpy 数组
    combined_samples = np.array(combined_samples)
    combined_labels = np.array(combined_labels)
    combined_samples = np.vstack(combined_samples)

    return combined_samples, combined_labels




def classify_wsi_and_vote_verbose(model, wsi_features, device):
    """
    对每个 WSI 的特征进行分类，并通过多数投票得到最终结果。
    打印每个 WSI 的分类结果和每个类别的计数。
    """
    model.eval()
    wsi_final_predictions = {}

    with torch.no_grad():
        for wsi_id, features in wsi_features.items():
            predictions = []
            for feature in features:
                feature = feature.to(device).unsqueeze(0)  # 增加 batch 维度
                outputs = model(feature)
                _, predicted = torch.max(outputs, 1)
                predictions.append(predicted.item())

            # 计数每个类别的结果
            prediction_counts = Counter(predictions)
            final_prediction = prediction_counts.most_common(1)[0][0]
            wsi_final_predictions[wsi_id] = final_prediction

            # 为输出创建格式化的字符串
            count_str = ", ".join(str(prediction_counts[i]) for i in range(5))  # 假设总共 5 类
            print(f"ID: {wsi_id}, Categories: ({count_str})")

    return wsi_final_predictions