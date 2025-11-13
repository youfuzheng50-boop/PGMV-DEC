from sklearn.cluster import KMeans
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import umap
from sklearn.manifold import TSNE
from matplotlib.colors import ListedColormap, Normalize

def cosine_similarity(X, centers):
    # 计算 X 的范数，并避免除以零
    X_norm = np.linalg.norm(X, axis=1, keepdims=True)
    X_norm = np.where(X_norm == 0, 1e-10, X_norm)  # 防止零除

    # 计算 centers 的范数
    centers_norm = np.linalg.norm(centers, axis=1, keepdims=True)
    centers_norm = np.where(centers_norm == 0, 1e-10, centers_norm)  # 防止零除

    # 进行点积并计算相似度
    dot_product = np.dot(X, centers.T)
    norm_product = np.dot(X_norm, centers_norm.T)

    # 计算余弦相似度
    similarities = dot_product / norm_product

    return similarities



class ConstrainedKMeans(KMeans):
    def __init__(self, n_clusters=8, max_iter=20, initial_centers=None, must_link=None, cannot_link=None, sample_weights=None, mode='centroid', prototype_indices=None, **kwargs):
        super().__init__(n_clusters=n_clusters, max_iter=max_iter, **kwargs)
        self.initial_centers = initial_centers
        self.must_link = must_link if must_link is not None else []
        self.cannot_link = cannot_link if cannot_link is not None else []
        self.sample_weights = sample_weights
        self.mode = mode
        self.prototype_indices = prototype_indices  # 使用样本索引指定原型

    def fit(self, X, y=None):
        if self.initial_centers is None:
            self.cluster_centers_ = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]
        else:
            self.cluster_centers_ = self.initial_centers

        if self.sample_weights is None:
            self.sample_weights = np.ones(X.shape[0])

        if self.must_link:
            prototype_weight = 10
            for i, j in self.must_link:
                self.sample_weights[i] = prototype_weight
                self.sample_weights[j] = prototype_weight

        # # 从数据集中提取原型
        # if self.prototype_indices is not None:
        #     prototypes = X[self.prototype_indices].squeeze()
        # else:
        #     raise ValueError("Prototype indices must be provided.")

        for _ in range(self.max_iter):
            if self.mode == 'c':
                similarities = cosine_similarity(X, self.cluster_centers_)
            # elif self.mode == 'p':
            #     similarities = cosine_similarity(X, prototypes)
            else:
                raise ValueError("Invalid mode specified.")

            self.labels_ = np.argmax(similarities, axis=1)
            self.cluster_similarity_ = similarities

            self._apply_constraints()

            for i in range(self.n_clusters):
                points = X[self.labels_ == i]
                weights = self.sample_weights[self.labels_ == i]
                if len(points) > 0:
                    self.cluster_centers_[i] = np.average(points, axis=0, weights=weights)

    def _apply_constraints(self):
        for i, j in self.must_link:
            if self.labels_[i] != self.labels_[j]:
                self.labels_[j] = self.labels_[i]

        for i, j in self.cannot_link:
            if self.labels_[i] == self.labels_[j]:
                for new_label in range(self.n_clusters):
                    if new_label != self.labels_[i]:
                        self.labels_[j] = new_label
                        break



class PrototypeKMeansClusterer:
    def __init__(self, n_clusters, p_features):
        """参数:
        n_clusters: 聚类数
        p_features: 原型特征的数组，格式为 [time_domain_prototypes, freq_domain_prototypes]
        """
        self.n_clusters = n_clusters
        self.p_features = p_features

    def select_prototype_features(self, class_label):
        """根据类标签选择 p_features 中的原型特征。"""
        if class_label not in [1, 2, 3, 4]:
            raise ValueError("class_label 必须是 1, 2, 3 或 4")

        start_idx = (class_label - 1) * 80
        end_idx = class_label * 80

        domain_prototypes = []
        for domain_feature in self.p_features:
            domain_prototypes.append(domain_feature[start_idx:end_idx])

        return domain_prototypes

    def fit(self, X_list, class_label):
        """将原型特征与样本数据合并，然后执行 KMeans 聚类。"""
        if not isinstance(X_list, list) or len(X_list) == 0:
            raise ValueError("X_list 必须是非空列表，包含一个或多个特征集")

        # 将 torch.Tensor 转换为 numpy，确保 X_list 的每个元素是 numpy 数组
        X_list = [x.cpu().numpy() if isinstance(x, torch.Tensor) else x for x in X_list]

        # 根据 class_label 选择原型特征
        prototypes_list = self.select_prototype_features(class_label)

        # 确保原型和数据域一致并合并
        combined_data = [
            np.vstack((prototypes_list[i], X_list[i])) if prototypes_list[i] is not None and X_list[
                i] is not None else None
            for i in range(len(X_list))
        ]

        labels, centers, similarities = [], [], []
        cancer_labels = []  # 初始化列表用于存储癌症标签
        normal_labels = []  # 初始化列表用于存储正常标签
        num_prototypes = prototypes_list[0].shape[0] if prototypes_list else 0
        cancer_indices = list(range(num_prototypes // 2))  # 癌症组织索引
        normal_indices = list(range(num_prototypes // 2, num_prototypes))  # 正常组织索引

        # must-link 和 cannot-link 约束
        must_link = [(i, j) for i in cancer_indices for j in cancer_indices if i < j] + \
                    [(i, j) for i in normal_indices for j in normal_indices if i < j]
        cannot_link = [(i, j) for i in cancer_indices for j in normal_indices]

        for data in combined_data:
            if data is None:
                # 如果 data 为 None，向结果列表中添加占位符
                labels.append(None)
                centers.append(None)
                similarities.append(None)
                cancer_labels.append(None)  # 如果需要，也为癌症标签添加占位符
                normal_labels.append(None)  # 如果需要，也为正常标签添加占位符
                continue  # 跳过当前循环，处理下一个 data
            kmeans = ConstrainedKMeans(n_clusters=self.n_clusters, must_link=must_link, cannot_link=cannot_link,
                                       mode='c', prototype_indices=[[0, 40]])
            kmeans.fit(data)

            sim = kmeans.cluster_similarity_
            num_samples = data.shape[0] - num_prototypes
            labels_tensor = torch.tensor(kmeans.labels_[num_prototypes:], dtype=torch.long)
            centers_tensor = torch.tensor(kmeans.cluster_centers_, dtype=torch.float)
            sims_tensor = torch.tensor(sim[num_prototypes:], dtype=torch.float)

            # 将单个标签添加到各自的列表中
            cancer_label = kmeans.labels_[50]  # 获取单个癌症标签
            cancer_labels.append(cancer_label)  # 将标签添加到列表

            nor_label = kmeans.labels_[0]  # 获取单个正常标签
            normal_labels.append(nor_label)  # 将标签添加到列表

            labels.append(labels_tensor)
            centers.append(centers_tensor)
            similarities.append(sims_tensor)

        # 返回所有值：包括标签列表而不是单个值
        return labels, centers, similarities, cancer_labels, normal_labels

    def plot_clusters(self, X_list, class_label, labels, centers):
        """
        可视化时域数据的聚类结果，并用不同的颜色标识不同簇，原型向量将在图中显示为特殊标记。

        参数:
        X_list: 包含两个 ndarray 的列表，其中第一个是时域的数据
        class_label: 用于选择原型特征的类标签
        labels: 聚类标签列表，第一个元素是时域数据的标签张量
        centers: 聚类中心列表，第一个元素是时域数据的聚类中心张量
        """
        # 检查并确保 X_list 是时域的格式
        time_data = X_list[0].cpu().numpy() if isinstance(X_list[0], torch.Tensor) else X_list[0]

        # 使用 UMAP 将数据降维到二维以便于可视化
        reducer = umap.UMAP(n_components=2)
        time_data_reduced = reducer.fit_transform(time_data)
        centers_reduced = reducer.transform(centers[0].cpu().numpy())  # 确保中心是 numpy 类型
        custom_colors = ['#C24E6C', '#6799A5']
        # 绘图
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(time_data_reduced[:, 0], time_data_reduced[:, 1],
                              c=labels[0].cpu().numpy(), cmap=ListedColormap(custom_colors), marker='o', alpha=0.7, s=7)
        plt.scatter(centers_reduced[:, 0], centers_reduced[:, 1],
                    c='black', marker='X', s=200, label='Cluster Centers')

        # 添加原型向量标记
        prototypes = self.select_prototype_features(class_label)[0]
        prototypes_reduced = reducer.transform(prototypes)

        # 前20个是癌症，用红色圆圈表示
        plt.scatter(prototypes_reduced[:40, 0], prototypes_reduced[:40, 1],
                    c='green', marker='s', s=50, label='Normal_prototype')

        # 后20个是正常组织，用绿色方块表示
        plt.scatter(prototypes_reduced[40:, 0], prototypes_reduced[40:, 1],
                    c='red', marker='s', s=50, label='Cancer_prototype')

        # 添加图例和标题
        # plt.legend()
        # 去掉坐标轴和边框
        plt.axis('off')
        plt.show()



def convert_to_numpy(tensor_or_list):
    if isinstance(tensor_or_list, torch.Tensor):
        return tensor_or_list.cpu().numpy()
    elif isinstance(tensor_or_list, list):
        return [tensor.cpu().numpy() if tensor is not None else None for tensor in tensor_or_list]
    else:
        raise TypeError("Input should be a Tensor or a list of Tensors")


def update_cluster_centers(embedding_list, n_clusters, emp_feature_list, class_indx):
    """
    更新聚类中心 (KMeans)

    参数:
    embedding_list: 样本嵌入列表
    n_clusters: 聚类数
    emp_feature_list: 原型特征列表
    class_label: 类标签用于选择原型特征

    返回:
    聚类中心和标签
    """
    # 转换 embeddings_list 和 emp_feature_list 为 NumPy 格式
    embeddings = convert_to_numpy(embedding_list)
    emp_features = convert_to_numpy(emp_feature_list)

    # 创建并使用 PrototypeKMeansClusterer
    clusterer = PrototypeKMeansClusterer(n_clusters=n_clusters, p_features=emp_features)
    labels, centers ,sim,cancer_label,normal_label= clusterer.fit(embeddings, class_label=class_indx)

    clusterer.plot_clusters(embeddings, class_indx, labels, centers)
    return labels, centers, sim,cancer_label,normal_label