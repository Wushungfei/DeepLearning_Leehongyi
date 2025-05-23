import torch.nn as nn


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