import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset

import numpy as np


# 定义MLP模型
class DeepMLPClassifier(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes=2, dropout_prob=0.5):
        """
        Deep MLP 模型，具有更多隐藏层
        input_size: 输入特征的维度
        hidden_sizes: 一个包含每层神经元数量的列表
        num_classes: 输出的类别数量
        dropout_prob: Dropout 概率
        """
        super(DeepMLPClassifier, self).__init__()

        layers = []

        # 输入层到第一层隐藏层
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(nn.BatchNorm1d(hidden_sizes[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_prob))

        # 增加更多隐藏层
        for i in range(1, len(hidden_sizes)):
            layers.append(nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]))
            layers.append(nn.BatchNorm1d(hidden_sizes[i]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_prob))

        # 输出层
        layers.append(nn.Linear(hidden_sizes[-1], num_classes))

        # 将所有层组合成一个 Sequential 模块
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)






class ContrastiveModel(nn.Module):
    def __init__(self, input_dim, projection_dim=1024):
        super(ContrastiveModel, self).__init__()

        # 增加层数，并且每一层都增大维度
        self.fc1 = nn.Linear(input_dim, 1024)  # 从输入维度到1024维
        self.fc2 = nn.Linear(1024, 512)  # 从1024维到512维
        self.fc3 = nn.Linear(512, 256)  # 从512维到256维
        self.fc4 = nn.Linear(256, projection_dim)  # 最终投影到目标维度

        # 加入 BatchNorm 和 Dropout 来正则化模型
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(256)

        self.dropout = nn.Dropout(0.3)  # Dropout 防止过拟合

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))  # 第一层 + BatchNorm + ReLU
        x = F.relu(self.bn2(self.fc2(x)))  # 第二层 + BatchNorm + ReLU
        x = F.relu(self.bn3(self.fc3(x)))  # 第三层 + BatchNorm + ReLU
        x = self.dropout(x)  # 加入 Dropout

        # x = F.normalize(self.fc4(x), dim=-1)  # 最后一层投影到单位球面
        return x


class BinaryClassifierMLP(nn.Module):
    def __init__(self, input_dim, num_heads=4):
        super(BinaryClassifierMLP, self).__init__()
        # 共享的隐藏层
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.relu = nn.ReLU()

        # 为每个类别创建一个输出头
        self.output_heads = nn.ModuleList([nn.Linear(64, 2) for _ in range(num_heads)])  # 2表示二分类

    def forward(self, x, head_idx):
        # 共享层
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))

        # 根据类别选择不同的输出头
        output = self.output_heads[head_idx](x)
        return output

#
def classify_intermediate_samples(intermediate_samples, model, device,head_idx, batch_size=512):
    if not intermediate_samples.any():  # 检查是否为空
        print(f"Warning: No intermediate samples for head_idx={head_idx}. Skipping this category.")
        return []  # 直接返回空列表
    model.eval()  # 确保模型在推理模式
    confidences = []
    predictions = []
    # combined_samples = np.concatenate(intermediate_samples, axis=0)  # 将所有样本纵向合并
    combined_samples = torch.tensor(intermediate_samples).float().to(device)
    total_samples = len(combined_samples)
    with torch.no_grad():  # 关闭梯度计算以加速推理
        for i in range(0, total_samples, batch_size):
            batch_samples = combined_samples[i:i + batch_size]
            # 确保每个样本为Tensor
            batch_samples = [torch.tensor(sample) if isinstance(sample, np.ndarray) else sample for sample in
                             batch_samples]

            batch_samples = torch.stack(batch_samples)  # 合并成一个 batch
            batch_samples = batch_samples.to(device)  # 将数据移到目标设备

            # 执行批量预测
            outputs = model(batch_samples, head_idx)
            batch_predictions = torch.argmax(outputs, dim=1)  # 根据任务需求可能需要不同处理
            conf_scores = torch.max(outputs, dim=1)[0]
            predictions.extend(batch_predictions.cpu().numpy())
            confidences.extend(conf_scores.cpu().numpy())
    return predictions,confidences


class FiveClassClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(FiveClassClassifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 1024),  # 第一层
            nn.ReLU(),
            nn.Dropout(0.5),  # 调整为0.5

            nn.Linear(1024, 512),  # 确保维度连续
            nn.ReLU(),
            nn.Dropout(0.4),  # 调整为0.4

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),  # 调整为0.3

            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, num_classes)  # 输出层
        )

    def forward(self, x):
        return self.fc(x)





class FiveClassClassifier_tcga(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(FiveClassClassifier_tcga, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),  # 第一层
            nn.ReLU(),
            nn.Dropout(0.6),  # 调整为0.5

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),  # 调整为0.3

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),  # 调整为0.3

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, num_classes)  # 输出层
            )

    def forward(self, x):
        return self.fc(x)


class MLPClassifier(nn.Module):
    def __init__(self, input_size, hidden_size,num_classes):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 1024)  # 第一层
        self.fc2 = nn.Linear(1024, 512)  # 第二层
        self.fc3 = nn.Linear(512, 256)  # 第三层
        self.fc4 = nn.Linear(256, hidden_size)  # 第四层
        self.fc5 = nn.Linear(hidden_size, num_classes)  # 输出层

        # Dropout层
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.4)
        self.dropout3 = nn.Dropout(0.3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)  # 第一层后 Dropout

        x = F.relu(self.fc2(x))
        x = self.dropout2(x)  # 第二层后 Dropout

        x = F.relu(self.fc3(x))
        x = self.dropout3(x)  # 第三层后 Dropout

        features = F.relu(self.fc4(x))  # 得到中间特征

        x = self.fc5(features)  # 输出层
        return features, x  # 返回中间特征和分类输出


#区分自编码器
class DiscriminativeAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim, num_classes):
        super(DiscriminativeAutoencoder, self).__init__()
        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
            nn.ReLU()
        )

        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.Linear(512, input_dim),
        )

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, num_classes),
        )

    def forward(self, x):
        z = self.encoder(x)  # 编码得到 latent 特征
        x_reconstructed = self.decoder(z)  # 重构输入
        class_logits = self.classifier(z)  # 分类预测
        embedding = self.encoder(x)
        return x_reconstructed, class_logits, embedding