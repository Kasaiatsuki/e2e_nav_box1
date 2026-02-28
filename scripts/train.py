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

# 入力画像の幅と高さ、および推論対象のウェイポイント数
IMAGE_WIDTH = 64
IMAGE_HEIGHT = 48
NUM_WAYPOINTS = 10


class E2EDataset(Dataset):
    """
    End-to-End学習用のPyTorchデータセットクラス。
    画像データ(RGBマスク)とこれに対応するパス(x, y座標)のセットを読み込み、
    学習に適したテンソル形式に変換して提供する。
    """
    def __init__(self, dataset_path: Path):
        self.dataset_path = dataset_path
        self.mask_images_dir = dataset_path / 'mask_images'
        self.path_dir = dataset_path / 'path'
        # マスク画像(.png)をソートしてリスト化し、対応するパスデータ(.csv)を取得しやすくする
        self.mask_files = sorted(list(self.mask_images_dir.glob('*.png')))

    def __len__(self) -> int:
        """データセットの総サンプル数を返す"""
        return len(self.mask_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        指定されたインデックスのデータ(画像テンソル, ウェイポイントテンソル)を返す。
        画像はバイナリマスク化、クロップ、リサイズを経て1chのテンソルにする。
        ウェイポイントは指定のスケールで正規化( -1.0 〜 1.0 等)する。
        """
        mask_file = self.mask_files[idx]
        csv_file = self.path_dir / f'{mask_file.stem}.csv'

        # OpenCVを用いて画像(BGR形式)を読み込み、RGB形式に変換
        image_bgr = cv2.imread(str(mask_file), cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # CSVファイルからウェイポイント(x, y)を読み込む
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            waypoints = [[float(row['x']), float(row['y'])] for row in reader]

        # ネットワークに入力する関心領域(ROI)のみをクロップ(x:40〜440)
        cropped_image = image_rgb[:, 40:440]
        # 学習用に指定サイズ(IMAGE_WIDTH x IMAGE_HEIGHT)へリサイズ(ニアレストネイバー補間またはバイリニア補間)
        resized_image = cv2.resize(cropped_image, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_LINEAR)

        # PyTorchで扱えるようにfloat32へ変換し、[0, 255]の値を[0, 1]に正規化
        image_normalized = resized_image.astype(np.float32) / 255.0
        # HWC (Height, Width, Channels) から CHW (Channels, Height, Width) の順に並べ替える
        image_tensor = torch.from_numpy(image_normalized).permute(2, 0, 1)

        # ウェイポイントのリストをテンソルに変換し、1次元に平坦化する
        waypoints_tensor = torch.tensor(waypoints, dtype=torch.float32).flatten()
        # x座標(インデックスが偶数)とy座標(インデックスが奇数)をそれぞれ正規化(例: -1.0 〜 1.0)する
        waypoints_tensor[0::2] = waypoints_tensor[0::2] / 5.0 - 1.0
        waypoints_tensor[1::2] = (waypoints_tensor[1::2] + 3.0) / 3.0 - 1.0
        """
        実際の距離が 0m の時 ⇒ 0 / 5 - 1 = -1.0
        実際の距離が 5m の時 ⇒ 5 / 5 - 1 = 0.0
        実際の距離が 10m の時 ⇒ 10 / 5 - 1 = 1.0
        waypointを0.5m刻みで10個取得しているのなら一番遠いwaypointは5秒後なのでこの正規化だと2m/sまでしか出せない。
        これは2m/sでデータ収集すればokということ？
        """
        return mask_tensor, waypoints_tensor

class Config:
    """
    学習に関する設定(ハイパーパラメータやパス)をYAMLファイルからロードし管理するクラス。
    """
    def __init__(self, config_path: Path, package_root: Path):
        self.package_root = package_root

        with open(config_path, 'r') as f:
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
    def __init__(self, dataset_path: Path, config: Config):
        self.config = config
        self.device = config.device

        # データセットをロードし、8割を学習用、2割を検証用に分割する
        dataset = E2EDataset(dataset_path)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

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
        self.model = Network(num_waypoints=NUM_WAYPOINTS).to(self.device)
        # オプティマイザ(最適化アルゴリズム)の設定: AdamW
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.learning_rate)
        # 損失関数の定義: 平均二乗誤差(MSE)
        self.mseloss = nn.MSELoss()
        # TensorBoardへログを出力するためのライター
        self.writer = SummaryWriter(log_dir=str(config.logs_dir))

        # 最良の検証ロスを保持しておく変数。初期値は無限大
        self.best_val_loss = float('inf')

        print(f'Using device: {self.device}')
        print(f'Train size: {len(train_dataset)}, Val size: {len(val_dataset)}')

    def validate(self) -> float:
        """
        検証データセットに対して順伝播を行い、ロスを計算する。
        学習(パラメータの更新)は行わない。
        """
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad(): # 勾配計算の無効化(メモリ節約、高速化)
            pbar = tqdm(self.val_loader, desc='Validation')
            for images, waypoints in pbar:
                # データをGPUへ転送
                images = images.to(self.device)
                waypoints = waypoints.to(self.device)

                # 順伝播で推論を実行し、損失を計算
                outputs = self.model(images)
                loss = self.mseloss(outputs, waypoints)

                # バッチのロスを蓄積
                total_loss += loss.item()
                pbar.set_postfix({'loss': f'{loss.item():.6f}'})

        # 全バッチの平均ロスを返す
        return total_loss / len(self.val_loader)

    def save_checkpoint(self, val_loss: float) -> None:
        """
        検証ロスが今までで最も良かった場合、モデルの重みを保存(Checkpointing)する。
        """
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            weight_path = self.config.weights_dir / self.config.weight_file
            
            # TorchScript形式(C++からも読み込み可能)でモデルを直列化(コンパイル)して保存する
            scripted_model = torch.jit.script(self.model)
            scripted_model.save(str(weight_path))
            print(f'Best model saved: {weight_path} (val_loss: {val_loss:.6f})')

    def train(self, epochs: int) -> None:
        """
        指定されたエポック数だけ学習と検証のサイクルを実行するメインループ。
        """
        for epoch in range(1, epochs + 1):
            self.model.train()
            total_train_loss = 0.0

            pbar = tqdm(self.train_loader, desc=f'Epoch {epoch} [Train]')
            for images, waypoints in pbar:
                images = images.to(self.device)
                waypoints = waypoints.to(self.device)

                # 勾配の初期化(これをしないと前回の微分の値が残ってしまう)
                self.optimizer.zero_grad()
                
                # 順伝播(推論)
                outputs = self.model(images)

                # 損失の計算と逆伝播(勾配の計算)
                loss = self.mseloss(outputs, waypoints)
                loss.backward()
                
                # オプティマイザによるパラメータ(重み)の更新
                self.optimizer.step()

                total_train_loss += loss.item()
                pbar.set_postfix({'loss': f'{loss.item():.6f}'})

            # 平均の訓練ロスと検証ロスを計算
            train_loss = total_train_loss / len(self.train_loader)
            val_loss = self.validate()

            # TensorBoardに損失をグラフ化するため記録する
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)

            print(f'Epoch [{epoch}/{epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')

            # より良いモデルであれば重みファイルを更新
            self.save_checkpoint(val_loss)

        self.writer.close()

def main() -> None:
    """コマンドラインから実行させる際のメインエントリポイント"""
    if len(sys.argv) != 2:
        print('Usage: python3 train.py <dataset_path>')
        sys.exit(1)

    dataset_path = Path(sys.argv[1])
    if not dataset_path.exists():
        print(f'Dataset path does not exist: {dataset_path}')
        sys.exit(1)

    # config.yaml(train.yaml)のパスを相対的に取得
    script_dir = Path(__file__).parent
    package_root = script_dir.parent
    config_path = package_root / 'config' / 'train.yaml'

    # 設定のロードとTrainerの初期化
    config = Config(config_path, package_root)
    trainer = Trainer(dataset_path, config)

    print(f'Starting training for {config.epochs} epochs')
    trainer.train(config.epochs)

    print('Training complete.')

if __name__ == '__main__':
    main()
