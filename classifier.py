# 加载数据
import random
from collections import Counter

import torch
from torch import optim, nn
from torch.utils.data import TensorDataset, DataLoader

from models.MLP import FiveClassClassifier


def balance_samples_downsample(selected_samples, selected_labels):
    """
    将样本数量多的类别下采样至与样本最少的类别数量相同。

    参数：
        selected_samples (torch.Tensor): 输入的样本，形状为 (n, 1024)。
        selected_labels (list): 样本对应的标签列表，长度为 n。

    返回：
        balanced_samples (torch.Tensor): 平衡后的样本，形状为 (m, 1024)，m 为平衡后的样本数量。
        balanced_labels (list): 平衡后的标签列表，长度为 m。
    """
    # 检查输入有效性
    if len(selected_labels) != selected_samples.shape[0]:
        raise ValueError("selected_samples 的行数必须与 selected_labels 的长度一致")

    # 统计每个类别的样本数量
    label_counts = Counter(selected_labels)
    min_count = min(label_counts.values())  # 找到样本最少的类别数量

    # 初始化存储平衡结果的列表
    balanced_samples = []
    balanced_labels = []

    # 平衡每个类别
    for label in label_counts.keys():
        # 获取当前类别的样本索引
        indices = [i for i, lbl in enumerate(selected_labels) if lbl == label]

        # 提取当前类别的样本
        class_samples = selected_samples[indices]

        # 对样本数量多的类别进行下采样
        if len(indices) > min_count:
            sampled_indices = random.sample(indices, k=min_count)
            class_samples = selected_samples[sampled_indices]

        # 添加平衡后的样本和对应的标签
        balanced_samples.append(class_samples)
        balanced_labels.extend([label] * min_count)

    # 将平衡后的样本合并成一个张量
    balanced_samples = torch.cat(balanced_samples, dim=0)

    return balanced_samples, balanced_labels

# 训练5分类模型
def train_five_class_model(five_class_model, dataloader, optimizer, epochs=40, patience=10):
    """
    训练五类模型，支持早停机制。

    Parameters:
    - five_class_model: 要训练的模型
    - dataloader: 提供数据的 Dataloader
    - optimizer: 优化器
    - epochs: 最大训练轮数
    - patience: 早停的容忍轮数
    """
    five_class_model.train()
    criterion = nn.CrossEntropyLoss()

    best_loss = float('inf')  # 初始化最小损失为正无穷
    trigger_times = 0  # 初始化没有改进的epoch计数

    for epoch in range(epochs):
        running_loss = 0.0
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = five_class_model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)  # 计算平均损失
        print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")

        # 检查是否需要保存模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(five_class_model.state_dict(),
                       "/home/idao/Zyf/models/oc_subtype_classficter/multi_view/samples_k3/fiveClassifier_aaa.pth")
            # print(f'Model saved with validation loss: {best_loss:.4f}')
            trigger_times = 0
        else:
            trigger_times += 1

        # 检查是否早停
        if trigger_times >= patience:
            print(f'Early stopping at epoch {epoch + 1}')
            break

    return five_class_model  # 返回训练后的模型

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
five_class_model = FiveClassClassifier(input_dim=1024, num_classes=6)
five_class_model.to(device)


data = torch.load("Pseudo-label samples/pseudo_labeled_dataset.pt")
selected_labels = data["features"]
selected_samples = data["labels"]

# 用 Counter 统计每个类的数量
label_counts = Counter(selected_labels)

# 打印每个类的数量
for label, count in label_counts.items():
    print(f"Label {label}: {count}")
print(len(selected_labels))
optimizer_2 = optim.Adam(five_class_model.parameters(), lr=0.0001)
balanced_samples, balanced_labels = balance_samples_downsample(selected_samples, selected_labels)

# print('平衡后样本数量：')
# label_counts = Counter(selected_labels )
# for label, count in label_counts.items():
#     print(f"Label {label}: {count}")
balanced_labels = torch.tensor(balanced_labels, dtype=torch.long)
# 创建平衡的数据集和数据加载器
# balanced_dataset = TensorDataset(balanced_samples, balanced_labels)
selected_labels = torch.tensor(balanced_labels, dtype=torch.long)
balanced_dataset = TensorDataset(balanced_samples, selected_labels)
balanced_dataloader = DataLoader(balanced_dataset, batch_size=8192, shuffle=True,num_workers=16)

# 训练五分类模型
train_five_class_model(five_class_model, balanced_dataloader, optimizer_2, epochs=300,patience=4)
torch.save(five_class_model.state_dict(), '/UBC_OC/fold_2/UBC_2.pth')