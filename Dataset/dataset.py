import os
import pandas as pd
# from torchvision.io import read_image
from torch.utils.data import Dataset,dataloader
from PIL import Image
import torch

class OC_subtype_Dataset(Dataset):
    def __init__(self,data_csv_file,img_dir,transform=None,target_transform=None):
        self.img_lables = pd.read_csv(data_csv_file,dtype=str)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_lables)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir,self.img_lables.iloc[idx,1])
        image = Image.open(img_path)
        label = self.img_lables.iloc[idx,1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label



class Binary_Dataset(Dataset):
    def __init__(self, class_0_features, class_1_features):
        # 将类别0的特征标记为0，类别1的特征标记为1
        self.features = torch.cat((torch.tensor(class_0_features, dtype=torch.float32),
                                   torch.tensor(class_1_features, dtype=torch.float32)), dim=0)
        self.labels = torch.cat((torch.zeros(len(class_0_features), dtype=torch.float32),
                                 torch.ones(len(class_1_features), dtype=torch.float32)), dim=0)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class BagDataset(Dataset):
    def __init__(self, bags):
        self.bags = bags  # 每个 bag 是一个包含多个实例特征的张量

    def __len__(self):
        return len(self.bags)

    def __getitem__(self, idx):
        # 返回 bag 和它的伪标签 (针对这个分类器来说，所有 bag 都是正类)
        return self.bags[idx], torch.ones(1)  # 这里伪标签为 1 表示正类







class SlideBagDataset(Dataset):
    def __init__(self, labels_df, label_mapping, pt_files_path):
        self.labels_df = labels_df
        self.label_mapping = label_mapping
        self.pt_files_path = pt_files_path

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        row = self.labels_df.iloc[idx]
        slide_id = row['slide_id']
        label = row['label']
        mapped_label = self.label_mapping[label]
        pt_file_path = os.path.join(self.pt_files_path, slide_id + ".pt")

        if os.path.exists(pt_file_path):
            bag_features = torch.load(pt_file_path)
            return (mapped_label, bag_features, slide_id)
        else:
            print(f"File {pt_file_path} not found!")
            return (None, None, slide_id)




class EmptyDataset(Dataset):
    def __len__(self):
        return 0

    def __getitem__(self, idx):
        raise IndexError("EmptyDataset has no data")