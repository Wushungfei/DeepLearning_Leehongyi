import numpy as np
import pandas as pd
import os
from torch.utils.data import Dataset
import torch
class BsurfaceDataset(Dataset):
    def __init__(self, root_dir, grid_size=(10, 10), train=True):
        self.samples = []
        self.train = train

        for file in os.listdir(root_dir):
            if file.endswith(".csv"):
                # 解析标签
                label_str = file.split('_')[0]
                label_map = {"sphere": 0, "cylinder": 1, "cone": 2
                             ,"plane":3 , "torus":4, "others":5}  # 示例映射
                label = label_map[label_str]

                # 读取数据
                df = pd.read_csv(os.path.join(root_dir, file))
                points = self._load_points(df, grid_size)

                # 存储样本
                self.samples.append((points, label))

    def _load_points(self, df, grid_size):
        """将CSV转换为网格点云"""
        array = np.zeros((grid_size[0], grid_size[1], 3))
        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                array[i, j] = df.iloc[i, 3 * j: 3 * j + 3].values
        return array.reshape(-1, 3)  # 展平为(N,3)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        points, label = self.samples[idx]

        # 训练时动态增强
        if self.train:
            # 随机旋转
            angle = np.random.uniform(0, 2 * np.pi)
            rot_mat = np.array([[np.cos(angle), -np.sin(angle), 0],
                                [np.sin(angle), np.cos(angle), 0],
                                [0, 0, 1]])
            points = points @ rot_mat

            # 添加噪声
            points += np.random.normal(0, 0.01, size=points.shape)

        # 归一化
        points = (points - points.min()) / (points.max() - points.min() + 1e-8)

        # return torch.FloatTensor(points), torch.LongTensor([label])
        return torch.FloatTensor(points), torch.tensor(label, dtype=torch.long)

if __name__ =="__main__":
    data = BsurfaceDataset("D:/GitProject/qy-cad-adapter-pro/testFile/surface/")
    print(data.__getitem__(0))
