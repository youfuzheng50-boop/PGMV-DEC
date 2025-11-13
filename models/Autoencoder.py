import torch
import torch.nn as nn
import torch.optim as optim
# from cuml import KMeans  # 使用cuML的KMeans
import numpy as np
import torch.nn.functional as F
from sklearn.cluster import KMeans

import cupy as cp
from sklearn.cluster import SpectralClustering

from models.k_means_sim import ClusterSelector



# 加载.pt文件数据
def load_data(file_path):
    data = torch.load(file_path,weights_only=True)
    return data







# def update_cluster_centers(embeddings, n_clusters):
#     # 把 embeddings 从 PyTorch Tensor 转换为 NumPy 数组
#     embeddings_np = embeddings.cpu().detach().numpy()
#
#     # 使用 SpectralClustering 进行 CPU 上的聚类
#     spectral_clustering = SpectralClustering(n_clusters=n_clusters, assign_labels='discretize', random_state=0)
#     spectral_clustering.fit(embeddings_np)
#
#     # 获取聚类标签，并转换为 CuPy 数组
#     labels = spectral_clustering.labels_
#     labels_cp = cp.asarray(labels)  # 将标签移至 GPU
#
#     # 将 embeddings 移至 GPU
#     embeddings_gpu = cp.asarray(embeddings_np)
#
#     # 计算每个聚类的中心
#     cluster_centers = []
#     for i in range(n_clusters):
#         # 在 GPU 上使用标签
#         cluster_indices = cp.where(labels_cp == i)[0]
#         cluster_center = cp.mean(embeddings_gpu[cluster_indices], axis=0).get()  # 计算并转移回 CPU
#         cluster_centers.append(cluster_center)
#
#     # 将 Python 列表转换为 PyTorch tensor 并使其与初始设备一致
#     return torch.tensor(cluster_centers, dtype=torch.float, device=embeddings.device)
