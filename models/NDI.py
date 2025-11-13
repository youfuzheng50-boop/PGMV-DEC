import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import pickle as pkl
import math
import numpy as np


class Build_NDI():
    def __init__(self, num_classifiers=4, length=40, tau=0.9, loss_weight=0.3, max_margin=0.4, lambda_max=1.0,
                 warmup_epochs=2, use_triplet_loss=False, use_proto_sim=False):
        self.NDI_queue_length = length
        self.num_classifiers = num_classifiers  # 二分类器的数量
        self.tau = tau
        self.alpha = loss_weight

        # 为每个二分类器初始化一个原型向量队列和对应的得分队列
        self.NDI = [[] for _ in range(self.num_classifiers)]
        self.NDI_scores = [[] for _ in range(self.num_classifiers)]

        # 动态调整的参数
        self.margin = 0.5
        self.lambda_max = lambda_max
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0  # 初始化当前 epoch
        self.max_margin = max_margin

        # 新增：是否使用 Triplet Loss 和 Prototype Similarity
        self.use_triplet_loss = use_triplet_loss
        self.use_proto_sim = use_proto_sim

    def set_loss_enabled(self, use_class_loss=True, use_triplet_loss=False, use_proto_sim=False):
        self.use_triplet_loss = use_triplet_loss
        self.use_proto_sim = use_proto_sim


    @torch.no_grad()
    def Update_NDI(self, classifier_idx, features, obj_scores, labels_per_im):
        """
        更新指定二分类器的原型向量队列，保留置信度低的困难样本，无论标签为良性或恶性。
        所有难分类样本都会更新到对应的原型2队列。
        """
        # 提取预测得分和类别索引
        value, idxes = torch.max(obj_scores, dim=1)

        # 选择置信度低的样本（即困难样本），不论其类别
        difficult_samples = value < self.tau  # 低于置信度阈值的样本被认为是困难样本

        # 根据难分类样本的掩码选择相应的 features 和 obj_scores
        features = features[difficult_samples]
        obj_scores = value[difficult_samples]

        # 将这些难分类样本直接更新到当前分类器的原型2队列
        for f, s in zip(features, obj_scores):
            if len(self.NDI[classifier_idx]) == self.NDI_queue_length:
                # 如果原型向量队列已满，找到最相似的特征进行更新
                NDI_feats = self.NDI[classifier_idx]
                cos_similarity = torch.cosine_similarity(f[None, :], torch.cat(NDI_feats).reshape(-1, f.shape[-1]))
                ind = torch.argmax(cos_similarity)

                # 加权更新最相似的特征
                feat_old = self.NDI[classifier_idx][ind]
                s_old = self.NDI_scores[classifier_idx][ind]
                ratio = (s / (s + s_old + 1e-4))

                f_new = ratio * f + (1 - ratio) * feat_old
                s_new = ratio * s + (1 - ratio) * s_old

                # 更新原型向量和对应的置信度
                self.NDI[classifier_idx][ind] = f_new
                self.NDI_scores[classifier_idx][ind] = s_new
            else:
                # 如果队列未满，直接加入新的原型向量
                self.NDI[classifier_idx].append(f)
                self.NDI_scores[classifier_idx].append(s)

    def get_prototype_queue(self, classifier_idx):
        """
        获取指定二分类器的原型向量队列
        classifier_idx: 二分类器的索引
        返回该二分类器的所有原型向量
        """
        if len(self.NDI[classifier_idx]) == 0:
            raise ValueError(f"No prototype vectors available for classifier {classifier_idx}")

        # 返回该分类器的所有原型向量
        prototype_queue = self.NDI[classifier_idx]
        return prototype_queue

    def combined_loss(self, anchor, positive, negative, prototype_queue, predictions, targets):
        # 确保 targets 是 LongTensor 类型
        targets = targets.long()  # 将 targets 转换为 LongTensor

        # 计算分类损失（交叉熵损失）
        classification_loss = F.cross_entropy(predictions, targets)

        # 初始化 Triplet Loss 和原型相似损失
        triplet_loss = torch.tensor(0.0)
        prototype_sim = torch.tensor(0.0)

        if self.use_triplet_loss:
            # 控制 margin 的最大值
            dynamic_margin = min(self.margin * (1.0 + self.current_epoch / self.warmup_epochs), self.max_margin)
            # 计算 Triplet Loss（使用余弦相似度）
            positive_sim = F.cosine_similarity(anchor, positive)
            negative_sim = F.cosine_similarity(anchor, negative)
            triplet_loss = torch.relu(negative_sim - positive_sim + dynamic_margin).mean()

        if self.use_proto_sim:
            # 计算所有原型向量的余弦相似度，并取平均
            prototype_sims = [F.cosine_similarity(anchor, prototype) for prototype in prototype_queue]
            prototype_sim = torch.stack(prototype_sims).mean()

        # 动态调整原型向量的权重 lambda
        lambda_weight = min((self.current_epoch / (self.warmup_epochs * 0.7)), 1.0) * self.lambda_max

        # 总损失=分类损失+Triplet Loss（如果启用）+原型损失（如果启用）
        total_loss = classification_loss + triplet_loss + lambda_weight * (1 - prototype_sim)

        return total_loss, classification_loss, triplet_loss, prototype_sim

    def confidence_penalty(self, obj_scores, penalty_weight=0.2):
        """
        计算高置信度样本的惩罚项，鼓励模型学习难样本的特征。

        obj_scores: 模型输出的类别概率分布
        penalty_weight: 惩罚项的权重，控制惩罚强度
        """
        # 计算置信度惩罚项
        penalty = -torch.sum(obj_scores * torch.log(1.0 - obj_scores + 1e-6), dim=1)

        # 加权惩罚项
        weighted_penalty = penalty_weight * penalty.mean()

        return weighted_penalty

    def get_prototype_anchor(self, class_idx):
        """
        从原型向量队列中提取指定类别的原型向量。
        class_idx: 要提取的类别的索引
        返回该类别的代表性原型向量（可以是队列中的最后一个，或者根据其他规则选择）
        """
        if len(self.NDI[class_idx]) == 0:
            raise ValueError(f"No prototype vectors available for class {class_idx}")

        # 你可以选择不同的策略来返回原型向量
        # 这里假设取最后一个存储的原型向量
        prototype_anchor = self.NDI[class_idx][-1]
        return prototype_anchor

    def set_epoch(self, epoch):
        """
        设置当前 epoch，主要用于动态调整原型向量损失的权重。
        """
        self.current_epoch = epoch

    @torch.no_grad()
    def Update_NDI_after_difficult_classification(self, classifier_idx, features, obj_scores):
        """
        使用来自困难样本的高置信度癌症样本对原型向量进行大幅度更新。
        """
        # 提取预测得分
        value, _ = torch.max(obj_scores, dim=1)

        # 仅保留置信度高于某个阈值的样本
        high_confidence_samples = value > self.tau  # 设置较高的置信度阈值

        # 使用高置信度的困难样本进行原型更新
        features = features[high_confidence_samples]
        obj_scores = value[high_confidence_samples]

        for f, s in zip(features, obj_scores):
            if len(self.NDI[classifier_idx]) == self.NDI_queue_length:
                # 如果队列已满，找到最相似的原型向量进行大幅更新
                NDI_feats = self.NDI[classifier_idx]
                cos_similarity = torch.cosine_similarity(f[None, :], torch.cat(NDI_feats).reshape(-1, f.shape[-1]))
                ind = torch.argmax(cos_similarity)

                # 大幅更新：这里设置一个较大的更新权重 alpha
                alpha = 0.8  # 训练后的更新权重较大
                feat_old = self.NDI[classifier_idx][ind]
                f_new = alpha * f + (1 - alpha) * feat_old

                # 更新原型向量
                self.NDI[classifier_idx][ind] = f_new
                self.NDI_scores[classifier_idx][ind] = s
            else:
                # 如果队列未满，直接加入新的原型
                self.NDI[classifier_idx].append(f)
                self.NDI_scores[classifier_idx].append(s)

    def handle_insufficient_samples(self, anchor, positive, negative):
        # 获取 anchor、positive、negative 中最小的样本数量
        min_samples = min(anchor.size(0), positive.size(0), negative.size(0))

        # 根据最小的样本数量裁剪 anchor、positive、negative
        anchor = anchor[:min_samples]
        positive = positive[:min_samples]
        negative = negative[:min_samples]

        return anchor, positive, negative
