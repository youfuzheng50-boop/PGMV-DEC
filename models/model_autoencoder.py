import torch.nn as nn
import torch



class MultiViewAutoencoder(nn.Module):
    def __init__(self, input_dims, latent_dim, hidden_dim=256):
        super(MultiViewAutoencoder, self).__init__()

        self.view_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(512, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, latent_dim),
                nn.BatchNorm1d(latent_dim),
                nn.ReLU()
            ) for input_dim in input_dims
        ])

        self.view_decoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(512, input_dim),
                nn.Sigmoid()  # 假设输出在0-1之间，根据实际需要调整
            ) for input_dim in input_dims
        ])

    def encode(self, *views):
        encoded_views = []
        for encoder, view in zip(self.view_encoders, views):
            if view is None:
                encoded_views.append(None)  # 如果输入为 None，返回 None
            else:
                # 输入为 0 的情况不处理，返回 None
                encoded_views.append(encoder(view))
        return encoded_views

    def decode(self, *latent_representations):
        decoded_views = []
        for decoder, latent in zip(self.view_decoders, latent_representations):
            if latent is None:
                decoded_views.append(None)  # 如果潜在表示为 None，返回 None
            elif latent.sum() != 0:
                decoded_views.append(decoder(latent))
            else:
                # 输入为 0 的情况不处理，返回 None
                decoded_views.append(None)
        return decoded_views

    def forward(self, *views):
        # 编码阶段
        latent_representations = self.encode(*views)
        # 解码阶段
        reconstructed_views = self.decode(*latent_representations)
        return reconstructed_views


def add_noise(data, noise_factor=0.2):
    # 生成与原始数据形状相同的高斯噪声
    noise = noise_factor * torch.randn_like(data)
    noisy_data = data + noise

    # 根据需要裁剪数据值
    noisy_data = torch.clamp(noisy_data, 0., 1.)
    return noisy_data


class SharedSpecificAutoencoder(nn.Module):
    def __init__(self, input_dims, shared_dim, specific_dim, hidden_dims=[512, 256, 128]):
        """
        共享 + 特定表示自编码器 (4 层结构)

        Args:
            input_dims (list): 每个视图的输入维度列表。
            shared_dim (int): 共享表示的维度。
            specific_dim (int): 特定表示的维度。
            hidden_dims (list): 隐藏层的维度列表，从高到低。
        """
        super(SharedSpecificAutoencoder, self).__init__()

        self.view_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dims[0]),
                nn.ReLU(),
                nn.Linear(hidden_dims[0], hidden_dims[1]),
                nn.ReLU(),
                nn.Linear(hidden_dims[1], hidden_dims[2]),
                nn.ReLU(),
                nn.Linear(hidden_dims[2], shared_dim + specific_dim)
            ) for input_dim in input_dims
        ])

        self.view_decoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(shared_dim + specific_dim, hidden_dims[2]),
                nn.ReLU(),
                nn.Linear(hidden_dims[2], hidden_dims[1]),
                nn.ReLU(),
                nn.Linear(hidden_dims[1], hidden_dims[0]),
                nn.ReLU(),
                nn.Linear(hidden_dims[0], input_dim)
            ) for input_dim in input_dims
        ])

        self.shared_dim = shared_dim
        self.specific_dim = specific_dim

    def encode(self, *views):
        """
        对每个视图进行编码，分别提取共享和特定表示。

        Args:
            views (list of Tensor): 每个视图的输入特征。

        Returns:
            shared_list (list of Tensor): 每个视图的共享表示。
            specific_list (list of Tensor): 每个视图的特定表示。
        """
        shared_list = []
        specific_list = []

        for encoder, view in zip(self.view_encoders, views):
            latent = encoder(view)
            shared, specific = torch.split(latent, [self.shared_dim, self.specific_dim], dim=1)
            shared_list.append(shared)
            specific_list.append(specific)

        return shared_list, specific_list

    def decode(self, shared_list, specific_list):
        """
        对共享和特定表示进行解码，重构每个视图的输入。

        Args:
            shared_list (list of Tensor): 每个视图的共享表示。
            specific_list (list of Tensor): 每个视图的特定表示。

        Returns:
            reconstructed_views (list of Tensor): 每个视图的重构结果。
        """
        reconstructed_views = []

        for decoder, shared, specific in zip(self.view_decoders, shared_list, specific_list):
            latent = torch.cat([shared, specific], dim=1)
            reconstructed_views.append(decoder(latent))

        return reconstructed_views

    def forward(self, *views):
        """
        前向传播：编码 + 解码

        Args:
            views (list of Tensor): 每个视图的输入特征。

        Returns:
            shared_list (list of Tensor): 每个视图的共享表示。
            specific_list (list of Tensor): 每个视图的特定表示。
            reconstructed_views (list of Tensor): 每个视图的重构结果。
        """
        shared_list, specific_list = self.encode(*views)
        reconstructed_views = self.decode(shared_list, specific_list)
        return shared_list, specific_list, reconstructed_views