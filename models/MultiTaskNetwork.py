import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiTaskNetwork(nn.Module):
    def __init__(self, input_dim=1024, shared_dim=512, hidden_dim=256):
        super(MultiTaskNetwork, self).__init__()

        # 共享特征层：两层全连接网络
        self.shared_fc1 = nn.Linear(input_dim, shared_dim)
        self.shared_fc2 = nn.Linear(shared_dim, hidden_dim)
        self.dropout = nn.Dropout(0.5)  # Dropout layer to avoid overfitting

        # Task-specific 分支网络（两层全连接网络，每个任务独立）
        self.task1_fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.task1_fc2 = nn.Linear(hidden_dim // 2, 1)

        self.task2_fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.task2_fc2 = nn.Linear(hidden_dim // 2, 1)

        self.task3_fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.task3_fc2 = nn.Linear(hidden_dim // 2, 1)

        self.task4_fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.task4_fc2 = nn.Linear(hidden_dim // 2, 1)

        # 激活函数
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 共享特征层：两层 MLP 加 ReLU 和 Dropout
        x = self.relu(self.shared_fc1(x))
        x = self.dropout(x)  # 添加 Dropout 层
        x = self.relu(self.shared_fc2(x))
        x = self.dropout(x)

        # Task-specific 分支网络
        # Task 1
        out1 = self.relu(self.task1_fc1(x))
        out1 = self.sigmoid(self.task1_fc2(out1))

        # Task 2
        out2 = self.relu(self.task2_fc1(x))
        out2 = self.sigmoid(self.task2_fc2(out2))

        # Task 3
        out3 = self.relu(self.task3_fc1(x))
        out3 = self.sigmoid(self.task3_fc2(out3))

        # Task 4
        out4 = self.relu(self.task4_fc1(x))
        out4 = self.sigmoid(self.task4_fc2(out4))

        return out1, out2, out3, out4

