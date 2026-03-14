import sys
import os
# パッケージのルートディレクトリを検索パスに追加（直接実行用）
package_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if package_root not in sys.path:
    sys.path.insert(0, package_root)

import yaml
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import cv2
import csv
from pathlib import Path
import numpy as np
from typing import Tuple
from tqdm import tqdm
from schedulefree import RAdamScheduleFree
from e2e_nav_box1.network import Network
from e2e_nav_box1.image_processor import ImageProcessor
from e2e_nav_box1.tensorboard_utils import TensorBoardLogger


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
        
        zero_velocity_count = 0
        kept_zero_velocity_count = 0
        
        for img_path in all_images:
            csv_file = self.angular_vel_dir / f'{img_path.stem}.csv'
            if csv_file.exists():
                # 角速度を読み込んでスキップ判定を行う
                with open(csv_file, 'r') as f:
                    try:
                        angular_z = float(f.read().strip())
                    except ValueError:
                        continue
                
                # 直線(angular_zがほぼ0)のデータを減らす処理
                if self.reduce_linner(angular_z):
                    self.image_files.append(img_path)
                    if abs(angular_z) < 1e-9:
                        kept_zero_velocity_count += 1
                elif abs(angular_z) < 1e-9:
                    zero_velocity_count += 1
        
        print(f"Dataset initialization:")
        print(f"  Total images found: {len(all_images)}")
        print(f"  Zero velocity images reduced: {zero_velocity_count}")
        print(f"  Zero velocity images kept: {kept_zero_velocity_count}")
        print(f"  Total images kept: {len(self.image_files)}")

    def reduce_linner(self, angular_z: float) -> bool:
        """
        直線(角速度がほぼ0)のデータを一定の確率で間引く。
        返り値: Trueの場合データを保持、Falseの場合スキップ
        """
        if abs(angular_z) < 1e-9:
            # 1/3の確率で保持 (2/3の確率でスキップ)
            return random.random() < 1/3
        return True

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

        from datetime import datetime
        # TensorBoardのログ保存先ディレクトリ (実行ごとにユニークなフォルダを作成)
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        self.logs_dir = package_root / 'runs' / current_time

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
        
        # 損失関数の定義: MSELoss
        self.mseloss = nn.MSELoss()

        # TensorBoardへログを出力するためのライター
        self.writer = TensorBoardLogger(log_dir=config.logs_dir)

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
                
                # 最初のエポックの最初のバッチだけ画像を記録する
                if batch_i == 0:
                    self.writer.add_images('Train/Images', images, epoch)

                images = images.to(self.device)
                targets = targets.to(self.device)

                # 勾配の初期化(これをしないと前回の微分の値が残ってしまう)
                self.optimizer.zero_grad()
                
                # 順伝播(推論)
                outputs = self.model(images)

                # 損失の計算
                loss = self.mseloss(outputs, targets)
                
                # 逆伝播(勾配の計算)
                loss.backward()
                
                # オプティマイザによるパラメータ(重み)の更新
                self.optimizer.step()

                # TensorBoardにイテレーションごとのロスを記録
                self.writer.add_scalar('Loss/train_iteration', loss.item(), iteration)

                total_train_loss += loss.item()
                pbar.set_postfix({'loss': f'{loss.item():.6f}'})

            # 検証 (Validation) ループ
            self.model.eval()
            total_val_loss = 0.0
            with torch.no_grad():
                for images, targets in self.val_loader:
                    images = images.to(self.device)
                    targets = targets.to(self.device)
                    outputs = self.model(images)
                    loss = self.mseloss(outputs, targets)
                    total_val_loss += loss.item()

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

