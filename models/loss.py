from os.path import basename
import cv2
from numpy import dtype
from tensorboard.compat.tensorflow_stub.dtypes import float32
from tqdm import tqdm
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.signal import butter, lfilter
import numpy as np

from sklearn.preprocessing import StandardScaler
from DCmodels.kmeans import PrototypeKMeansClusterer, cosine_similarity
from sklearn.decomposition import PCA




# def cluster_loss(embeddings_list, cluster_centers_list, labels_list,device):
#
#     total_loss = 0  # 初始化总损失
#     alpha = 1  # 样本损失的权重
#     beta = 1  # 原型相似性损失的权重
#     gamma = 0.5 # 聚类中心相似性损失的权重
#     delta = 1  # 癌症和正常原型相似性损失的权重
#     for embeddings, cluster_centers, labels in zip(embeddings_list, cluster_centers_list, labels_list):
#         cluster_centers = cluster_centers.to(device)
#         # 计算每个样本与其所属聚类中心之间的余弦相似度
#         sample_similarities = F.cosine_similarity(embeddings.unsqueeze(1), cluster_centers.unsqueeze(0), dim=2)
#         sample_center_similarities = sample_similarities[range(sample_similarities.size(0)), labels]
#         # 最大化相似度，使用1减去相似度得到损失
#         sample_loss = 1 - torch.mean(sample_center_similarities)
#
#         # # 获取前20个样本（癌症）的原型
#         cancer_prototypes = embeddings[:20, :]
#         # 计算所有癌症原型之间的相似度
#         cancer_similarity_matrix = F.cosine_similarity(cancer_prototypes.unsqueeze(1), cancer_prototypes.unsqueeze(0),
#                                                        dim=2)
#         cancer_similarity_values = cancer_similarity_matrix.triu(diagonal=1)
#         cancer_prototype_loss = 1 - torch.mean(cancer_similarity_values)
#
#         # 获取后20个样本（正常）的原型
#         normal_prototypes = embeddings[20:40, :]
#         # 计算所有正常原型之间的相似度
#         normal_similarity_matrix = F.cosine_similarity(normal_prototypes.unsqueeze(1), normal_prototypes.unsqueeze(0),
#                                                        dim=2)
#         normal_similarity_values = normal_similarity_matrix.triu(diagonal=1)
#         normal_prototype_loss = 1 - torch.mean(normal_similarity_values)
#
#         # 计算癌症和正常原型之间的相似度并最小化
#         inter_class_similarity_matrix = F.cosine_similarity(cancer_prototypes.unsqueeze(1),
#                                                 normal_prototypes.unsqueeze(0), dim=2)
#         inter_class_prototype_loss = torch.mean(inter_class_similarity_matrix)
#
#         # 总的原型相似性损失
#         prototype_similarity_loss = (
#                                       normal_prototype_loss + cancer_prototype_loss + delta*inter_class_prototype_loss)
#
#         # 计算聚类中心之间的余弦相似度
#         center_similarities = F.cosine_similarity(cluster_centers.unsqueeze(1), cluster_centers.unsqueeze(0), dim=2)
#         triu_mask = torch.triu(torch.ones(center_similarities.size(), device=center_similarities.device),
#                                diagonal=1).bool()
#         center_similarities = center_similarities[triu_mask]
#         center_separation_loss = torch.mean(center_similarities)
#
#         # 累加每个域的损失
#         domain_loss = beta * prototype_similarity_loss + gamma*center_separation_loss + alpha*sample_loss
#         total_loss += domain_loss
#
#     return total_loss


def cluster_loss(embeddings_list, cluster_centers_list, labels_list, device):
    total_loss = 0  # 初始化总损失
    alpha = 8 # 样本损失的权重
    beta = 1  # 原型相似性损失的权重
    gamma = 2  # 聚类中心相似性损失的权重
    delta = 1  # 癌症和正常原型相似性损失的权重

    # 将所有的损失计算并行化
    for embeddings, cluster_centers, labels in zip(embeddings_list, cluster_centers_list, labels_list):
        cluster_centers = cluster_centers.to(device)

        # 计算每个样本与其所属聚类中心之间的余弦相似度
        sample_similarities = F.cosine_similarity(embeddings.unsqueeze(1), cluster_centers.unsqueeze(0), dim=2)

        # 最大化相似度，使用1减去相似度得到损失
        sample_center_similarities = sample_similarities[range(sample_similarities.size(0)), labels]
        sample_loss = 1 - torch.mean(sample_center_similarities)

        # 获取前20个样本（癌症）的原型
        cancer_prototypes = embeddings[:20, :]
        # 获取后20个样本（正常）的原型
        normal_prototypes = embeddings[20:40, :]

        # 计算原型之间的相似度
        all_prototypes = torch.cat((cancer_prototypes, normal_prototypes), dim=0)
        similarity_matrix = F.cosine_similarity(all_prototypes.unsqueeze(1), all_prototypes.unsqueeze(0), dim=2)

        # 计算癌症和正常原型损失
        cancer_similarity_values = similarity_matrix[:20, :20].triu(diagonal=1)
        normal_similarity_values = similarity_matrix[20:, 20:].triu(diagonal=1)
        cancer_prototype_loss = 1 - torch.mean(cancer_similarity_values)
        normal_prototype_loss = 1 - torch.mean(normal_similarity_values)

        # 计算癌症和正常原型之间的相似度
        inter_class_prototype_loss = torch.mean(similarity_matrix[:20, 20:])

        # 总的原型相似性损失
        prototype_similarity_loss = (
                2*normal_prototype_loss + 2*cancer_prototype_loss + delta * inter_class_prototype_loss
        )

        # 计算聚类中心之间的余弦相似度
        center_similarities = F.cosine_similarity(cluster_centers.unsqueeze(1), cluster_centers.unsqueeze(0), dim=2)
        triu_mask = torch.triu(torch.ones(center_similarities.size(), device=center_similarities.device),
                               diagonal=1).bool()
        center_separation_loss = torch.mean(center_similarities[triu_mask])

        # 累加每个域的损失
        domain_loss = beta * prototype_similarity_loss + gamma * center_separation_loss + alpha * sample_loss
        total_loss += domain_loss

    return total_loss

def multiview_cluster_loss(embeddings_list, cluster_centers_list, labels_list, device):
    total_loss = 0  # 初始化总损失
    beta = 1  # 原型相似性损失的权重
    gamma = 1  # 聚类中心相似性损失的权重
    delta = 1  # 癌症和正常原型相似性损失的权重
    theta = 1  # 多视图相似性矩阵一致性损失的权重

    # # 多视图损失（循环外计算）
    # view_1, view_2 = embeddings_list  # 提取两个视图
    # sim_matrix_1 = F.cosine_similarity(view_1.unsqueeze(1), view_1.unsqueeze(0), dim=2)
    # sim_matrix_2 = F.cosine_similarity(view_2.unsqueeze(1), view_2.unsqueeze(0), dim=2)
    # multi_view_loss = torch.norm(sim_matrix_1 - sim_matrix_2, p='fro')  # Frobenius 范数作为一致性损失

    # 计算原型和聚类中心的损失
    for embeddings, cluster_centers, labels in zip(embeddings_list, cluster_centers_list, labels_list):
        cluster_centers = cluster_centers.to(device)

        # # 原型相似性损失计算
        # cancer_prototypes = embeddings[:20, :]
        # normal_prototypes = embeddings[20:40, :]
        # all_prototypes = torch.cat((cancer_prototypes, normal_prototypes), dim=0)
        # similarity_matrix = F.cosine_similarity(all_prototypes.unsqueeze(1), all_prototypes.unsqueeze(0), dim=2)
        #
        # cancer_similarity_values = similarity_matrix[:20, :20].triu(diagonal=1)
        # normal_similarity_values = similarity_matrix[20:, 20:].triu(diagonal=1)
        # cancer_prototype_loss = 1 - torch.mean(cancer_similarity_values)
        # normal_prototype_loss = 1 - torch.mean(normal_similarity_values)
        # inter_class_prototype_loss = torch.mean(similarity_matrix[:20, 20:])
        # prototype_similarity_loss = (
        #     normal_prototype_loss + cancer_prototype_loss + delta * inter_class_prototype_loss
        # )

        # 聚类中心相似性损失计算
        center_similarities = F.cosine_similarity(cluster_centers.unsqueeze(1), cluster_centers.unsqueeze(0), dim=2)
        triu_mask = torch.triu(torch.ones(center_similarities.size(), device=center_similarities.device),
                               diagonal=1).bool()
        center_separation_loss = torch.mean(center_similarities[triu_mask])

        # 累加每个域的损失
        domain_loss = (
            gamma * center_separation_loss
        )
        total_loss += domain_loss

    # # 最终总损失
    # total_loss += theta * multi_view_loss
    return total_loss