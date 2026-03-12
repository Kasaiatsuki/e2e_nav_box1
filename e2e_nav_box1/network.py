#!/usr/bin/env python3

import torch
import torch.nn as nn

class Network(nn.Module):
    """
    End-to-Endプランニング用の改善されたCNNモデル。
    畳み込み層を増やして特徴抽出を強化し、全結合層のパラメータを削減した構成。
    """
    def __init__(self):
        super(Network, self).__init__()

        # 入力: (3, 144, 256)
        
        # Conv層1: 5x5, stride 2, 32ch -> 出力: (32, 70, 126)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Conv層2: 3x3, stride 2, 64ch -> 出力: (64, 34, 62)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Conv層3: 3x3, stride 2, 128ch -> 出力: (128, 16, 30)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Conv層4: 3x3, stride 2, 256ch -> 出力: (256, 8, 15)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        # Conv層5: 3x3, stride 2, 256ch -> 出力: (256, 4, 8)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        
        # 平坦化
        self.flatten = nn.Flatten()
        
        # 全結合層: 実行時に動的に入力サイズを決定する
        self.fc1 = None  # forward内で初期化
        self.dropout = nn.Dropout(0.5)
        
        # 出力層: 角速度
        self.fc2 = nn.Linear(256, 1)

        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        順伝播処理。
        各畳み込み層の後にBNとReLUを適用。
        """
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.relu(self.bn5(self.conv5(x)))
        
        x = self.flatten(x)
        
        # FC1層を動的に初期化（最初の推論時のみ）
        if self.fc1 is None:
            n_size = x.shape[1]
            self.fc1 = nn.Linear(n_size, 256).to(x.device)
            print(f"[Network] Dynamic FC1 input size: {n_size}")
        
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x
