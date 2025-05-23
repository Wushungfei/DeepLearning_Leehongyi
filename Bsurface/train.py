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
###### 目的:输入B曲面的采样点信息和对应的曲面标签，完成B曲面的分类任务

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
class Classifier(nn.Module):
    def __init__(self, num_classes=3):
        super(Classifier, self).__init__()

        self.mlp = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )

        self.pool = nn.AdaptiveMaxPool1d(1)

        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # 输入形状: (B, N, 3)
        x = x.permute(0, 2, 1)  # (B, 3, N)
        x = self.mlp(x)  # (B, 1024, N)
        x = self.pool(x)  # (B, 1024, 1)
        x = x.view(x.size(0), -1)
        return self.fc(x)
def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Classifier(num_classes=6).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    best_val_acc = 0.0
    train_losses, val_accs = [], []
    # Data = DataProcess.BsurfaceDataset("D:/GitProject/qy-cad-adapter-pro/testFile/surface/")
    # train_dataset , test_dataset = Data.Divide(0.8)
    # train_loader= DataLoader(train_dataset, batch_size=32, shuffle=True)
    # test_loader = DataLoader(test_dataset, batch_size=32)
    for epoch in range(100):
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
            for points, labels in test_loader:
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

        print(f"Epoch {epoch + 1}/100 | Loss: {epoch_loss:.4f} | Val Acc: {val_acc:.4f}")
    return train_losses, val_accs

if __name__ == "__main__":
    # 初始化数据集
    full_dataset = BsurfaceDataset("D:/GitProject/qy-cad-adapter-pro/testFile/surface/", train=True)
    # 划分训练/验证/测试
    train_size = int(0.8 * len(full_dataset))
    val_size = (len(full_dataset) - train_size) // 2
    test_size = len(full_dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size]
    )

    # 创建DataLoader
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)
    train_losses, val_accs = train_model()
    # 绘制曲线
    plot_training_curves(train_losses, val_accs)