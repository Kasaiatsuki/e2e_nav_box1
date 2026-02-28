#!/usr/bin/env python3

import torch
import torch.nn as nn

class Network(nn.Module):
    """
    End-to-Endプランニング用の畳み込みニューラルネットワーク(CNN)モデル。
    入力として2次元の画像(あるいはそれに準ずるデータ)を受け取り、
    指定された数のウェイポイント(x, y座標)を出力する。
    """
    def __init__(self, num_waypoints: int = 10):
        super(Network, self).__init__()

        # Conv層1: 入力チャンネル1、出力チャンネル32、カーネルサイズ8、ストライド4
        # 画像の特徴量を大まかに抽出する最初の層
        self.conv1 = nn.Conv2d(1, 32, kernel_size=8, stride=4)
        
        # Conv層2: 入力チャンネル32、出力チャンネル64、カーネルサイズ3、ストライド2
        # より中程度のレベルの特徴量を抽出する層
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
        
        # Conv層3: 入力チャンネル64、出力チャンネル64、カーネルサイズ3、ストライド1
        # より高次で詳細な特徴量を抽出する層
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # 全結合層(Fully Connected Layer)の前に2次元のテンソルを1次元に平坦化(Flatten)するための層
        self.flatten = nn.Flatten()
        
        # 全結合層1: 入力数960 (直前の特徴量マップを平坦化したサイズ)、出力数512
        self.fc1 = nn.Linear(960, 512)
        
        # 全結合層2(出力層): 最終的に出力したいウェイポイント数 × 2 (xとy座標のため)
        self.fc2 = nn.Linear(512, num_waypoints * 2)

        # 活性化関数: ReLU (Rectified Linear Unit)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        順伝播(Forward pass)の処理を定義する。
        各畳み込み層と全結合層(fc1)の後にReLU活性化関数を適用する。
        最終層(fc2)では活性化関数を通さずに直接出力する。
        """
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.flatten(x)        # 畳み込みから全結合層へ繋ぐための平坦化
        x = self.relu(self.fc1(x))
        x = self.fc2(x)            # 出力 (x, y が1次元配列として並んだもの)

        return x
