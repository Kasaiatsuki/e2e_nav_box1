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
from e2e_nav_box1.image_processor import ImageProcessor
from e2e_nav_box1.shannon_surprise import ShannonSurprise


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
        self.processor = ImageProcessor()
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

        # データ拡張と前処理
        image_tensor, adjusted_angular_z = self.processor.augment_and_preprocess(image_bgr, angular_z, idx)

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
        self.num_bins = config_dict.get('num_bins', 7)     # シャノンサプライズ用のビン数 (デフォルト7)
        self.min_ang = config_dict.get('min_ang', -1.0)    # 角速度の最小範囲
        self.max_ang = config_dict.get('max_ang', 1.0)     # 角速度の最大範囲
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

        # 順伝播(推論)
        self.model = Network().to(self.device)
        # オプティマイザ(最適化アルゴリズム)の設定: AdamW
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.learning_rate)
        
        # --- シャノンサプライズ用の事前準備 ---
        print("Calculating steering angle distribution for Shannon Surprise...")
        all_angles = []
        for img_path in dataset.image_files:
            csv_file = dataset.angular_vel_dir / f'{img_path.stem}.csv'
            with open(csv_file, 'r') as f:
                all_angles.append(float(f.read().strip()))
        
        all_angles_tensor = torch.tensor(all_angles, device=self.device)
        self.surprise_handler = ShannonSurprise(
            all_angles_tensor, 
            config.num_bins, 
            min_val=config.min_ang, 
            max_val=config.max_ang
        )
        print(f"Angle Range: [{config.min_ang}, {config.max_ang}], Bins: {config.num_bins}")
        print(f"Distribution: {self.surprise_handler.bin_probs.tolist()}")
        # ------------------------------------

        # 損失関数の定義: 各サンプルの重み付けを可能にするため reduction='none' に設定
        self.mseloss = nn.MSELoss(reduction='none')
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
            for batch_i, (images, targets) in enumerate(pbar):
                iteration = (epoch - 1) * len(self.train_loader) + batch_i
                images = images.to(self.device)
                targets = targets.to(self.device)

                # 勾配の初期化(これをしないと前回の微分の値が残ってしまう)
                self.optimizer.zero_grad()
                
                # 順伝播(推論)
                outputs = self.model(images)

                # 損失の計算とシャノンサプライズによる重み付け
                loss = self.mseloss(outputs, targets)
                
                # サプライズ（重み）の計算と適用
                # targets は [batch_size, 1] なので角度のみ取り出す
                angles = targets.squeeze(1)
                shannon_weights = self.surprise_handler.compute_weights(angles)
                
                # 重みを各サンプルのLossに掛けて合計する
                weighted_loss = (loss.squeeze(1) * shannon_weights).mean()
                
                # 逆伝播(勾配の計算)
                weighted_loss.backward()
                
                # オプティマイザによるパラメータ(重み)の更新
                self.optimizer.step()

                # TensorBoardにイテレーションごとのロスを記録
                self.writer.add_scalar('Loss/train_iteration', weighted_loss.item(), iteration)

                total_train_loss += weighted_loss.item()
                pbar.set_postfix({'loss': f'{weighted_loss.item():.6f}'})

            # 検証 (Validation) ループ
            self.model.eval()
            total_val_loss = 0.0
            with torch.no_grad():
                for images, targets in self.val_loader:
                    images = images.to(self.device)
                    targets = targets.to(self.device)
                    outputs = self.model(images)
                    loss = self.mseloss(outputs, targets)
                    
                    # 検証時も重みを考慮する場合
                    angles = targets.squeeze(1)
                    shannon_weights = self.surprise_handler.compute_weights(angles)
                    weighted_loss = (loss.squeeze(1) * shannon_weights).mean()
                    total_val_loss += weighted_loss.item()

            train_loss = total_train_loss / len(self.train_loader)
            val_loss = total_val_loss / len(self.val_loader)

            # TensorBoardに訓練ロスと検証ロスを記録
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)

            print(f'Epoch [{epoch}/{epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')

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

