import numpy as np
import pandas as pd
import os
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
from pyecharts.charts import Line
from pyecharts import options as opts
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 必须导入才能启用3D绘图
###### 目的:输入B曲面的采样点信息和对应的曲面标签，完成B曲面的分类任务
label_map = {"sphere": 0, "cylinder": 1, "cone": 2
    , "plane": 3, "torus": 4, "others": 5,
       "Bsurf":6,"Fsurf":7,"Csurf":8,"Dsurf":9,"Esurf":10}  # 示例映射
class BsurfaceDataset(Dataset):
    def visualize_sample(self, idx):
        """
        可视化指定索引的点云样本。

        :param idx: 要可视化的样本索引
        """
        points = self.samples[idx]
        label = self.labels[idx]
        # 转换为 NumPy 并展平为 (N, 3)
        points_np = points.transpose(1, 2, 0)  # (H, W, 3)
        x = points_np[:, :, 0].flatten()
        y = points_np[:, :, 1].flatten()
        z = points_np[:, :, 2].flatten()
        # 获取类别名称
        label_names = {v: k for k, v in label_map.items()}
        label_name = label_names.get(label, "Unknown")
        # 创建 3D 图形
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        # 绘制散点图
        scatter = ax.scatter(x, y, z, c=z, cmap='viridis', s=10)
        # 设置标签和标题
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(f"Sample {idx} - Class: {label_name}")
        # 添加颜色条
        ax.set_box_aspect([1, 1, 1])  # 统一 XYZ 比例
        plt.colorbar(scatter, ax=ax, shrink=0.6, label='Z Value')
        # 显示图像
        plt.tight_layout()
        plt.show()
    def __init__(self,  data_dir, is_test=False):
        self.samples = []
        self.labels = []
        self.data_dir = data_dir
        # self.transform = transform
        self.is_test = is_test
        for file in os.listdir(data_dir):
            if file.endswith(".csv"):
                # 解析标签
                label_str = file.split('_')[0]
                label = label_map[label_str]
                # print( file,label_str,label )
                self.labels.append(label)
                # 读取数据
                grid_size =(25,25)
                df = pd.read_csv(os.path.join(data_dir, file),
                                 header=None,  # 如果CSV没有表头
                                 dtype=np.float32,  # 指定数据类型减少内存
                                 engine='c' , # 使用C引擎加速
                                 usecols = range(grid_size[1]*3)
                                 )
                points = self._load_points(df, grid_size=grid_size)
                # 存储样本
                self.samples.append(points)
                # print(label,points)
                # self.visualize_sample(len(self.samples)-1)

    def _load_points(self, df, grid_size):
        """将CSV转换为网格点云"""
        # array = np.zeros((grid_size[0], grid_size[1], 3))
        # for i in range(grid_size[0]):
        #     for j in range(grid_size[1]):
        #         array[i, j] = df.iloc[i, 3 * j: 3 * j + 3].values
        # return array.transpose(2, 0, 1) # 展平为(N,3)
        # 将整个DataFrame转换为numpy数组
        data = df.to_numpy(dtype=np.float32)
        # 直接重塑形状 (假设数据排列顺序正确)
        data = data.reshape(grid_size[0], grid_size[1], 3)
        return data.transpose(2, 0, 1)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        points = self.samples[idx]
        label = self.labels[idx]
        ### 数据增强
        # if self.test:
        #
        # 归一化
        # points = (points - points.min()) / (points.max() - points.min() + 1e-8)
        return torch.FloatTensor(points), torch.tensor(label, dtype=torch.long)

class BasicBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)
    def forward(self, x):
        residual = x
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        return torch.relu(x)

class Classifier(nn.Module):
    def __init__(self, num_classes):
        super(Classifier, self).__init__()
        self.conv_layers = nn.Sequential(
            # 输入 (3, 10, 10)
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            BasicBlock(64),
            BasicBlock(64),
            nn.MaxPool2d(kernel_size=2, stride=2),  # -> (64, 5, 5)

            BasicBlock(64),
            BasicBlock(64),
            nn.AdaptiveAvgPool2d((1, 1))  # -> (64, 1, 1)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
        # super(Classifier, self).__init__()
        # self.conv_layers = nn.Sequential(
        #     # 第一个卷积块
        #     nn.Conv2d(3, 32, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(inplace=True),
        #     # nn.MaxPool2d(kernel_size=2, stride=2),
        #     # nn.Dropout(p=0.02),
        #
        #     # 第二个卷积块
        #     nn.Conv2d(32, 64, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        #     nn.Dropout(p=0.5),
        #
        #     # 第三个卷积块
        #     nn.Conv2d(64, 128, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(inplace=True),
        # )
        #
        # # 自适应平均池化层
        # self.adaptive_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        #
        # # 全连接层
        # self.classifier = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(128 * 1 * 1, 512),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(p=0.02),
        #     nn.Linear(512, 256),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(p=0.02),
        #     nn.Linear(256, num_classes)
        # )

    def forward(self, x):
        x = self.conv_layers(x)
        # x = self.adaptive_avg_pool(x)
        x = self.classifier(x)
        return x
def plot_training_curves(train_losses, val_accs):
    """绘制训练损失和验证准确率曲线"""
    epochs = list(range(1, len(train_losses) + 1))  # X轴数据：训练轮次
    # 绘制损失曲线
    line_loss = (
        Line()
        .add_xaxis(epochs)
        .add_yaxis(
            series_name="Training Loss",
            y_axis=train_losses,
            label_opts=opts.LabelOpts(is_show=False),  # 不显示数据点标签
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(title="训练损失曲线", pos_left="center"),
            tooltip_opts=opts.TooltipOpts(trigger="axis"),
            xaxis_opts=opts.AxisOpts(name="Epoch", boundary_gap=False),
            yaxis_opts=opts.AxisOpts(name="Loss", type_="value"),
            toolbox_opts=opts.ToolboxOpts(is_show=True),  # 显示工具箱
        )
    )
    # 绘制准确率曲线（转换为百分比）
    line_acc = (
        Line()
        .add_xaxis(epochs)
        .add_yaxis(
            series_name="Validation Accuracy",
            y_axis=[acc * 100 for acc in val_accs],
            label_opts=opts.LabelOpts(is_show=False),
            linestyle_opts=opts.LineStyleOpts(color="green"),  # 自定义颜色
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(title="验证准确率曲线", pos_left="center"),
            xaxis_opts=opts.AxisOpts(name="Epoch"),
            yaxis_opts=opts.AxisOpts(
                name="Accuracy (%)",
                axislabel_opts=opts.LabelOpts(formatter="{value}%")  # 添加百分号
            ),
        )
    )
    # 保存为HTML文件
    line_loss.render("training_loss.html")
    line_acc.render("validation_accuracy.html")
    print("曲线图已保存至当前目录")

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Classifier(num_classes=len(label_map)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01,
                                              steps_per_epoch=len(train_loader),
                                              epochs=100)
    best_val_acc = 0.0
    train_losses, val_accs = [], []

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for points, labels in val_loader:
            points = points.to(device)
            labels = labels.to(device)

            outputs = model(points)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print("训练前:",correct / total)
    N  = 100
    for epoch in range(N):
        # 训练阶段
        model.train()
        running_loss = 0.0
        for points, labels in train_loader:
            points = points.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(points)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * points.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)

        # 验证阶段
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for points, labels in val_loader:
                points = points.to(device)
                labels = labels.to(device)

                outputs = model(points)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = correct / total
        val_accs.append(val_acc)

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")

        # 学习率调整
        scheduler.step()

        print(f"Epoch {epoch + 1}/",N, f"| Loss: {epoch_loss:.4f} | Val Acc: {val_acc:.4f}")
    return train_losses, val_accs


def visualize_point_cloud_by_class(dataset, label_map, num_samples_per_class=5):
    # 获取第一个样本
    for i  in range(len(dataset)):
        points, label = dataset[i]  # 返回的是 Tensor (3, 10, 10)
        # 转换为 NumPy 并展平为 (100, 3)
        points_np = points.numpy().transpose(1, 2, 0)  # -> (10, 10, 3)
        x = points_np[:, :, 0].flatten()
        y = points_np[:, :, 1].flatten()
        z = points_np[:, :, 2].flatten()
        # 打印标签信息
        label_names = {v: k for k, v in label_map.items()}
        print(f"Label: {label.item()} => '{label_names[label.item()]}'")
        # 创建 3D 图形
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        # 绘制散点图
        scatter = ax.scatter(x, y, z, c=z, cmap='viridis', s=20)
        # 设置标签和标题
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(f"First Sample: {label_names[label.item()]}")
        # 添加颜色条
        plt.colorbar(scatter, ax=ax, shrink=0.6, label='Z Value')
        # 显示图像
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    print("开始训练")
    # 初始化数据集
    full_dataset = BsurfaceDataset("D:/GitProject/qy-cad-adapter-pro/testFile/surface/")
    # 划分训练/验证/测试
    train_size = int(0.8 * len(full_dataset))
    val_size = (len(full_dataset) - train_size) // 2
    test_size = len(full_dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size]
    )
    print(len(train_dataset),len(val_dataset),len(test_dataset))
    # 创建DataLoader
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)
    print(train_loader.__len__(), val_loader.__len__(),test_loader.__len__())
    visualize_point_cloud_by_class(train_dataset, label_map, num_samples_per_class=5)
    train_losses, val_accs = train_model()
    # # # 绘制曲线
    plot_training_curves(train_losses, val_accs)