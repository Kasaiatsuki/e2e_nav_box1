#!/usr/bin/env python3

import sys
import yaml
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import cv2
import csv
from pathlib import Path
import numpy as np
from typing import Tuple
from tqdm import tqdm
from schedulefree import RAdamScheduleFree
from network import Network

# 入力画像の幅と高さ (1280x720の1/10)
IMAGE_WIDTH = 128
IMAGE_HEIGHT = 72

class E2EDataset(Dataset):
    """
    End-to-End学習用のPyTorchデータセットクラス。
    画像データ(RGBマスク)とこれに対応するパス(x, y座標)のセットを読み込み、
    学習に適したテンソル形式に変換して提供する。
    """
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        self.images_dir = dataset_dir / 'images'
        self.angular_vel_dir = dataset_dir / 'angular_vel'
        # 画像(.png)をソートしてリスト化し、対応する角速度データ(.csv)を取得しやすくする
        self.image_files = sorted(list(self.images_dir.glob('*.png')))

    def __len__(self) -> int:
        """データセットの総サンプル数を返す"""
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        指定されたインデックスのデータ(画像テンソル, 角速度テンソル)を返す。
        画像はバイナリマスク化、クロップ、リサイズを経て1chのテンソルにする。
        """
        image_file = self.image_files[idx]
        csv_file = self.angular_vel_dir / f'{image_file.stem}.csv'

        # OpenCVを用いて画像(BGR形式)を読み込む
        image_bgr = cv2.imread(str(image_file), cv2.IMREAD_COLOR)

        # CSVファイルから角速度(angular.z)を読み込む
        with open(csv_file, 'r') as f:
            # 中身の数値を直接読み取ってfloat変換
            angular_z = [float(f.read().strip())]

        # 画像のクロップ（トリミング）を廃止し、全体をそのまま指定サイズ(IMAGE_WIDTH x IMAGE_HEIGHT)へリサイズ
        resized_image = cv2.resize(image_bgr, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_LINEAR)

        # PyTorchで扱えるようにfloat32へ変換し、[0, 255]の値を[0, 1]に正規化
        image_normalized = resized_image.astype(np.float32) / 255.0
        # HWC (Height, Width, Channels) から CHW (Channels, Height, Width) の順に並べ替える
        image_tensor = torch.from_numpy(image_normalized).permute(2, 0, 1)

        return image_tensor, torch.tensor(angular_z, dtype=torch.float32)

class Config:
    """
    学習に関する設定(ハイパーパラメータやパス)をYAMLファイルからロードし管理するクラス。
    """
    def __init__(self, config_file, package_root):
        self.package_root = package_root

        with open(config_file, 'r') as f:
            config_dict = yaml.safe_load(f)

        self.epochs = config_dict['epochs']                # エポック数
        self.batch_size = config_dict['batch_size']        # バッチサイズ
        self.learning_rate = config_dict['learning_rate']  # 学習率
        self.num_workers = config_dict['num_workers']      # データロード時のプロセス数
        self.weight_file = config_dict['weight_file']      # 保存する重みファイル名
        """
        epochs: 200
        batch_size: 8
        learning_rate: 0.0002
        num_workers: 2
        weight_file: "e2e_model.pt"
        """

        # モデルの重みを保存するディレクトリを作成
        self.weights_dir = package_root / 'weights'
        self.weights_dir.mkdir(exist_ok=True)

        # TensorBoardのログ保存先ディレクトリ
        self.logs_dir = package_root / 'runs'

        # 学習を実行するデバイスの設定(GPU固定)
        self.device = torch.device('cuda')

class Trainer:
    """
    モデルの学習、検証(Validation)、重みの保存を管理・実行するクラス。
    """
    def __init__(self, dataset_dir, config):
        self.config = config
        self.device = config.device

        # データセットをロードし、学習用のデータをそのまま検証用にも使う
        dataset = E2EDataset(dataset_dir)
        train_size =  len(dataset)
        val_size =  len(dataset)
        train_dataset = dataset
        val_dataset = dataset

        # 学習用・検証用のDataLoaderを作成(バッチ提供、シャッフル、マルチプロセスなど)
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers
        )

        # モデルの生成と、指定の計算デバイス(GPU)への転送
        self.model = Network().to(self.device)
        # オプティマイザ(最適化アルゴリズム)の設定: AdamW
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.learning_rate)
        # 損失関数の定義: 平均二乗誤差(MSE)
        self.mseloss = nn.MSELoss()
        # TensorBoardへログを出力するためのライター
        self.writer = SummaryWriter(log_dir=str(config.logs_dir))

        print(f'Using device: {self.device}')
        print(f'Train size: {len(train_dataset)}')


    def train(self, epochs: int) -> None:
        """
        指定されたエポック数だけ学習と検証のサイクルを実行するメインループ。
        """
        for epoch in range(1, epochs + 1):
            self.model.train()
            total_train_loss = 0.0

            pbar = tqdm(self.train_loader, desc=f'Epoch {epoch} [Train]')
            for images, targets in pbar:
                images = images.to(self.device)
                targets = targets.to(self.device)

                # 勾配の初期化(これをしないと前回の微分の値が残ってしまう)
                self.optimizer.zero_grad()
                
                # 順伝播(推論)
                outputs = self.model(images)

                # 損失の計算と逆伝播(勾配の計算)
                loss = self.mseloss(outputs, targets)
                loss.backward()
                
                # オプティマイザによるパラメータ(重み)の更新
                self.optimizer.step()

                total_train_loss += loss.item()
                pbar.set_postfix({'loss': f'{loss.item():.6f}'})

            # 平均の訓練ロスと検証ロスを計算
            train_loss = total_train_loss / len(self.train_loader)

            # TensorBoardに損失をグラフ化するため記録する
            self.writer.add_scalar('Loss/train', train_loss, epoch)

            print(f'Epoch [{epoch}/{epochs}], Train Loss: {train_loss:.6f}')

        # ループ(for)を抜け、全エポックの学習が完全に終わった後に1回だけモデルを保存する
        weight_file_dest = self.config.weights_dir / self.config.weight_file
        scripted_model = torch.jit.script(self.model)
        scripted_model.save(str(weight_file_dest))
        print(f'Model saved to: {weight_file_dest}')
        
        self.writer.close()
def main() -> None:
    """コマンドラインから実行させる際のメインエントリポイント"""
    if len(sys.argv) != 2:
        print('Usage: python3 train.py <dataset_dir>')
        sys.exit(1)

    dataset_dir = Path(sys.argv[1])
    if not dataset_dir.exists():
        print(f'Dataset dir does not exist: {dataset_dir}')
        sys.exit(1)

    # config.yaml(train.yaml)のパスを相対的に取得
    script_dir = Path(__file__).parent
    package_root = script_dir.parent
    config_file = package_root / 'config' / 'train.yaml'

    # 設定のロードとTrainerの初期化
    config = Config(config_file, package_root)
    trainer = Trainer(dataset_dir, config)

    print(f'Starting training for {config.epochs} epochs')
    trainer.train(config.epochs)

    print('Training complete.')

if __name__ == '__main__':
    main()

