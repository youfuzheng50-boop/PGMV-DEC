from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import torch

# # 自定义KMeans类，重写fit方法
# class WeightedKMeans(KMeans):
#     def __init__(self, n_clusters=8, max_iter=300, initial_centers=None, **kwargs):
#         super().__init__(n_clusters=n_clusters, max_iter=max_iter, **kwargs)
#         self.initial_centers = initial_centers
#
#     def fit(self, X, y=None):
#         if self.initial_centers is None:
#             self.cluster_centers_ = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]
#         else:
#             self.cluster_centers_ = self.initial_centers
#
#         for _ in range(self.max_iter):
#             similarities = cosine_similarity(X, self.cluster_centers_)
#             self.labels_ = np.argmax(similarities, axis=1)  # 选择相似度最大的作为标签
#             self.cluster_similarity_ = similarities  # 存储每个样本与其聚类中心的相似度
#             for i in range(self.n_clusters):
#                 points = X[self.labels_ == i]
#                 if len(points) > 0:
#                     self.cluster_centers_[i] = points.mean(axis=0)
#
#         return self
#

# 定义距离计算函数
def cosine_similarity(X, centers):
    similarities = np.zeros((X.shape[0], centers.shape[0]))
    for i, center in enumerate(centers):
        similarities[:, i] = np.dot(X, center) / (np.linalg.norm(X, axis=1) * np.linalg.norm(center))
    return similarities


class ConstrainedKMeans(KMeans):
    def __init__(self, n_clusters=8, max_iter=20, initial_centers=None, must_link=None, cannot_link=None, sample_weights=None, **kwargs):
        super().__init__(n_clusters=n_clusters, max_iter=max_iter, **kwargs)
        self.initial_centers = initial_centers
        self.must_link = must_link if must_link is not None and len(must_link) > 0 else []  # Must-link 约束
        self.cannot_link = cannot_link if cannot_link is not None and len(cannot_link) > 0 else []  # Cannot-link 约束
        self.sample_weights = sample_weights  # 样本权重

    def fit(self, X, y=None):
        """
        根据 must-link 约束和 cannot-link 约束执行聚类。

        参数:
        - X: 输入的样本数据矩阵
        - y: 标签（可选）
        """
        # 初始化聚类中心
        if self.initial_centers is None:
            self.cluster_centers_ = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]
        else:
            self.cluster_centers_ = self.initial_centers

        # 如果没有提供样本权重，默认所有样本权重为1
        if self.sample_weights is None:
            self.sample_weights = np.ones(X.shape[0])

        # 如果有 must-link 约束，给 must-link 的样本加权
        if self.must_link:
            prototype_weight = 10 # 设定 must-link 样本的权重
            for i, j in self.must_link:
                self.sample_weights[i] = prototype_weight  # 为 must-link 的第一个样本加权
                self.sample_weights[j] = prototype_weight  # 为 must-link 的第二个样本加权

        for _ in range(self.max_iter):
            # 计算每个样本与聚类中心的相似度
            similarities = cosine_similarity(X, self.cluster_centers_)
            self.labels_ = np.argmax(similarities, axis=1)  # 选择相似度最大的作为标签
            self.cluster_similarity_ = similarities

            # 强制执行 must-link 和 cannot-link 约束
            self._apply_constraints()

            # 更新聚类中心，使用加权平均
            for i in range(self.n_clusters):
                points = X[self.labels_ == i]
                weights = self.sample_weights[self.labels_ == i]  # 对应样本的权重
                if len(points) > 0:
                    # 使用加权平均更新质心
                    self.cluster_centers_[i] = torch.tensor(np.average(points, axis=0, weights=weights))

        return self

    def _apply_constraints(self):
        """应用 must-link 和 cannot-link 约束."""
        # Must-link 约束：确保 must-link 中的样本拥有相同的标签
        for i, j in self.must_link:
            if self.labels_[i] != self.labels_[j]:
                # 如果两个样本标签不一致，我们强制将其中一个的标签设为相同
                self.labels_[j] = self.labels_[i]

        # Cannot-link 约束：确保 cannot-link 中的样本拥有不同的标签
        for i, j in self.cannot_link:
            if self.labels_[i] == self.labels_[j]:
                # 如果两个样本标签相同，强制调整其中一个的标签
                for new_label in range(self.n_clusters):
                    if new_label != self.labels_[i]:
                        self.labels_[j] = new_label
                        break


class ClusterSelector:
    def __init__(self, features, p_features, n_clusters=3, threshold=0.01, use_prototype=True):
        """
        初始化 ClusterSelector 类。

        参数:
        features: 样本特征矩阵
        p_features: 原型特征向量组
        n_clusters: 聚类数量，默认3
        threshold: 距离阈值，默认0.01
        use_prototype: 是否使用原型向量进行聚类，默认 True
        """
        # 添加调试信息
        if features is None:
            print("ERROR: 'features' is None during ClusterSelector Initialization")
        self.most_similar_cluster = None
        self.normal_label = None
        self.intermediate_label = None
        self.features = features
        self.p_features = p_features
        self.n_clusters = n_clusters
        self.threshold = threshold
        self.use_prototype = use_prototype  # 是否使用原型
        self.labels = None
        self.final_centers = None
        self.cluster_similarity = None
        self.must_link_indices = list(range(40)) if use_prototype else []  # 只有使用原型时才需要 must-link
        self.must_link_projections = None
        self.features_2d = None  # 存储降维后的 features

    def select_prototype_features(self, class_label):
        """
        根据类标签选择 p_features 中的原型特征。

        参数:
        class_label: 类标签 (1, 2, 3, 4)

        返回:
        选择后的原型特征
        """
        if class_label == 1:
            return self.p_features[:40]
        elif class_label == 2:
            return self.p_features[40:80]
        elif class_label == 3:
            return self.p_features[80:120]
        elif class_label == 4:
            return self.p_features[120:160]
        else:
            raise ValueError("class_label 必须是 1, 2, 3 或 4")

    def fit(self, class_label):
        """
        根据指定的类标签执行聚类。

        参数:
        class_label: 类标签 (1, 2, 3, 4)
        """
        if self.use_prototype:
            selected_p_features = self.select_prototype_features(class_label)

            # 如果 features 是 GPU 上的 PyTorch Tensor，则将其转换为 CPU 上的 NumPy 数组
            if isinstance(self.features, torch.Tensor):
                self.features = self.features.detach().cpu().numpy()

            # 如果 selected_p_features 是 GPU 上的 PyTorch Tensor，则将其转换为 CPU 上的 NumPy 数组
            if isinstance(selected_p_features, torch.Tensor):
                selected_p_features = selected_p_features.detach().cpu().numpy()

            # 合并 p_features 和 features，作为输入的完整特征矩阵
            combined_vertical = np.vstack((selected_p_features, self.features))
            del selected_p_features

            # 设置 must-link 约束
            must_link_indices = list(range(40))  # 根据 selected_p_features 调整 must-link 范围
            must_link = list(set(tuple(pair) for pair in [(i, j) for i in must_link_indices for j in must_link_indices if i != j]))

            # 打印 must-link 约束
            # print("Must-link constraints:", must_link)

            # 使用自定义的 ConstrainedKMeans 进行聚类
            kmeans = ConstrainedKMeans(n_clusters=self.n_clusters, must_link=must_link)
            kmeans.fit(combined_vertical)

            # 存储聚类结果
            self.final_centers = kmeans.cluster_centers_
            self.labels = kmeans.labels_
            self.cluster_similarity = kmeans.cluster_similarity_
            # 记录 must-link 投影后的结果
            pca = PCA(n_components=2)
            combined_2d = pca.fit_transform(combined_vertical)
            del combined_vertical
            # self.must_link_projections = combined_2d[:len(selected_p_features)]  # 记录 p_features 投影
            # self.features_2d = combined_2d[len(selected_p_features):]
            # self.centers_2d = pca.transform(self.final_centers)  # 在相同的 PCA 空间中对聚类中心进行投影# 记录降维后的 features 投影
        else:
            # 不使用原型向量时，直接对 features 进行聚类
            kmeans = ConstrainedKMeans(n_clusters=self.n_clusters)
            kmeans.fit(self.features)

            # 存储聚类结果
            self.final_centers = kmeans.cluster_centers_
            self.labels = kmeans.labels_

            # 对 features 进行降维并存储
            pca = PCA(n_components=2)
            self.features_2d = pca.fit_transform(self.features)
            self.centers_2d = pca.transform(self.final_centers)

    def get_cluster_labels(self):
        """
        返回聚类结果标签。
        """
        if self.use_prototype:
            return self.labels[40:]  # 去除掉前 40 个原型向量的标签
        else:
            return self.labels  # 如果不使用原型，则直接返回标签

    def select_samples(self, class_label):
        """
        根据聚类结果选择癌症样本、中间样本和正常样本。
        参数:
        class_label: 类标签 (1, 2, 3, 4)
        返回:
        cancer_selected_samples, intermediate_selected_samples, normal_selected_samples
        """
        if self.labels is None or self.final_centers is None:
            raise ValueError("请先执行 fit() 方法来进行聚类。")

        # 检查原型向量被分配到的类
        prototype_labels = self.labels[0] if self.use_prototype else None
        prototype_center = self.final_centers[prototype_labels].reshape(1, -1) if prototype_labels is not None else None
        class_index = cosine_similarity(prototype_center, self.final_centers) if prototype_labels is not None else None

        # 获取相似度的排序索引
        sorted_indices = np.argsort(class_index) if class_index is not None else None

        # 根据相似度排序结果，确定每个集群的标签
        intermediate_label = sorted_indices[0][1] if sorted_indices is not None else None
        normal_label = sorted_indices[0][0] if sorted_indices is not None else None
        most_similar_cluster = prototype_labels if prototype_labels is not None else None
        self.normal_label = normal_label
        self.intermediate_label = intermediate_label
        self.most_similar_cluster = most_similar_cluster

        # 提取 features 中的分类结果
        labels = self.get_cluster_labels()
        all_similarities = self.cluster_similarity
        cancer_similarity_threshold = 0.95
        normal_similarity_threshold = 0.93
        # 根据相似度阈值过滤样本
        cancer_selected_samples = self.features[
            (labels == most_similar_cluster) &
            (all_similarities[range(len(labels)), most_similar_cluster] >= cancer_similarity_threshold)
            ]
        # intermediate_selected_samples = self.features[
        #     (labels == intermediate_label) &
        #     (all_similarities[range(len(labels)), intermediate_label] >= similarity_threshold)
        #     ]
        normal_selected_samples = self.features[
            (labels == normal_label) &
            (all_similarities[range(len(labels)), normal_label] >= normal_similarity_threshold)
            ]
        # cancer_selected_samples = self.features[labels == most_similar_cluster] if most_similar_cluster is not None else None
        intermediate_selected_samples = self.features[labels == intermediate_label] if intermediate_label is not None else None
        # normal_selected_samples = self.features[labels == normal_label] if normal_label is not None else None

        # 返回三个类别的样本
        return cancer_selected_samples, intermediate_selected_samples, normal_selected_samples

    def plot_clusters(self):
        """
        绘制聚类的结果并标记 must-link 的点，以及聚类中心。
        """
        if self.labels is None:
            raise ValueError("请先执行 fit() 方法来进行聚类。")

        # 当使用原型向量时，根据原型向量的分配情况来确定类的身份（癌症、中间性、良性）
        if self.use_prototype:
            # 通过 select_samples 函数获取每个类别的样本
            cancer_selected_samples, intermediate_selected_samples, normal_selected_samples = self.select_samples(
                class_label=4)

            # 获取 select_samples 确定的类别标签
            intermediate_label, normal_label, most_similar_cluster = self.intermediate_label, self.normal_label, self.most_similar_cluster

            # 构建标签到颜色的映射
            label_color_mapping = {
                most_similar_cluster: '#46307E',  # 紫色（癌症）
                intermediate_label: '#1F988B',  # 蓝绿色（中间性）
                normal_label: '#E6E419'  # 黄色（良性）
            }

            labels = self.get_cluster_labels()
            # 根据标签映射颜色
            colors = [label_color_mapping.get(label, '#cccccc') for label in labels]  # 如果有未映射的标签，用灰色

        else:
            # 不使用原型向量时，使用默认的 colormap
            labels = self.get_cluster_labels()
            colors = labels  # 直接使用标签值映射到默认的 colormap

        # 使用在 fit 中已经降维好的数据进行绘图
        plt.figure(figsize=(10, 8))

        # 如果不使用原型向量，则用默认的 colormap
        if self.use_prototype:
            # 绘制聚类结果的散点图，并给散点图添加标签以便图例显示
            scatter = plt.scatter(self.features_2d[:, 0], self.features_2d[:, 1], c=colors, s=50)
        else:
            # 使用默认的 colormap 绘制
            scatter = plt.scatter(self.features_2d[:, 0], self.features_2d[:, 1], c=colors, cmap='viridis', s=50)

        # 标记 must-link 的点，使用在 fit 中记录的 must_link_projections（如果使用了原型）
        if self.use_prototype and self.must_link_projections is not None:
            plt.scatter(self.must_link_projections[:, 0], self.must_link_projections[:, 1],
                        c='red', marker='x', s=100, label='Must-link Points')

        # 标记聚类中心，直接使用 fit 中已经降维好的 centers_2d
        plt.scatter(self.centers_2d[:, 0], self.centers_2d[:, 1],
                    c='red', marker='*', s=500, edgecolor='white', label='Cluster Centers')

        if self.use_prototype:
            # 手动创建每个类别的图例项
            plt.scatter([], [], c='#46307E', label='Cancerous Samples')  # 紫色
            plt.scatter([], [], c='#1F988B', label='Intermediate Samples')  # 蓝绿色
            plt.scatter([], [], c='#E6E419', label='Normal Samples')  # 黄色

        # 添加图例和标题
        plt.legend(loc='upper right')
        plt.title(
            'Cluster Scatter Plot with Must-Link Points and Centers' if self.use_prototype else 'Cluster Scatter Plot')
        plt.colorbar(scatter)
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')

        # 显示图形
        plt.show()