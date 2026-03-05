#!/usr/bin/env python3

import sys
import yaml
import random
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
from e2e_nav_box1.network import Network

# 学習・推論の際の画像サイズ（HD720: 1280x720 からクロップして作成）
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 360

# 水平シフトクロップ拡張のパラメータ
# 0.0を2つ入れることで中央クロップ（拡張なし）の確率を上げる
SHIFT_SIGNS = [-2.0, -1.0, 0.0, 0.0, 1.0, 2.0]
SHIFT_VEL_OFFSET = 0.15  # 1段階あたりの角速度補正量 [rad/s]

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
        # 画像(.png)と対応する録画データ(.csv)が両方揃っているものだけをリスト化する
        all_images = sorted(list(self.images_dir.glob('*.png')))
        self.image_files = []
        for img_path in all_images:
            csv_file = self.angular_vel_dir / f'{img_path.stem}.csv'
            if csv_file.exists():
                self.image_files.append(img_path)

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
            angular_z = float(f.read().strip())

        # --- 水平シフトクロップ拡張 ---
        # 縦方向: 720→IMAGE_HEIGHT(360)にリサイズ（画素を全て保持）
        # 横方向: 1280のまま保持し、シフトクロップのみ適用
        h, w = image_bgr.shape[:2]
        shift_sign = random.choice(SHIFT_SIGNS)

        if w >= IMAGE_WIDTH:
            # Step1: 縦方向のみリサイズ (1280x720 → 1280x360)
            height_resized = cv2.resize(image_bgr, (w, IMAGE_HEIGHT), interpolation=cv2.INTER_LINEAR)
            # Step2: 横方向をシフトして640幅でクロップ (1280x360 → 640x360)
            max_x_shift = w - IMAGE_WIDTH
            center_x = max_x_shift // 2
            x_offset = int((shift_sign / 2.0) * center_x)
            x_start = max(0, min(center_x + x_offset, max_x_shift))
            cropped = height_resized[:, x_start:x_start + IMAGE_WIDTH]
        else:
            # 元画像がターゲットより小さい場合はリサイズのみ
            cropped = cv2.resize(image_bgr, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_LINEAR)
            shift_sign = 0.0

        # ずれた量に比例して角速度を補正（右にずれた画像→左に戻る指令を追加）
        adjusted_angular_z = angular_z + shift_sign * SHIFT_VEL_OFFSET

        # PyTorchで扱えるようにfloat32へ変換し、[0, 1]に正規化
        image_normalized = cropped.astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_normalized).permute(2, 0, 1)

        return image_tensor, torch.tensor([adjusted_angular_z], dtype=torch.float32)

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
    script_dir = Path(__file__).resolve().parent
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

