import cv2
import random
import torch
import numpy as np
from typing import Tuple

class ImageProcessor:
    """
    画像処理とデータ拡張（Data Augmentation）を管理するクラス。
    学習時(train.py)と推論時(inference_node.py)で共通の画像変換を行う。
    """
    def __init__(self):
        
        # 水平平行移動シフト拡張のパラメータ
        # ピクセル単位でシフト量を指定し、推論（shift=0）と一貫した変換を使う
        # 水平平行移動シフト拡張のパラメータ (256x144解像度に合わせて 128->256 の2倍スケール)
        self.shift_pixels = [-24, -12, 0, 0, 12, 24]
        self.shift_vel_per_pixel = 0.15 / 12.0

    def preprocess_for_inference(self, image_bgr: np.ndarray) -> torch.Tensor:
        """
        推論前の画像処理。
        リサイズ、正規化、およびCHW形式への変換を行う。
        """
        image_resized = cv2.resize(image_bgr, (256, 144))
        return self._to_tensor(image_resized)

    def augment_and_preprocess(self, image_bgr: np.ndarray, angular_z: float, idx: int) -> Tuple[torch.Tensor, float]:
        """
        学習用のデータ拡張と前処理。
        左右反転、水平シフト、環境光処理、リサイズ、正規化を行う。
        """
        image = image_bgr.copy()
        adjusted_angular_z = angular_z

        # Step2: 左右反転（サンプルのインデックスに基づいて1:1で適用）
        if idx % 2 == 1:
            image = cv2.flip(image, 1)
            adjusted_angular_z = -adjusted_angular_z

        # Step3: 水平平行移動シフト
        shift_px = random.choice(self.shift_pixels)
        if shift_px != 0:
            # シフト適用前に目標解像度に合わせたサイズ情報を取得（またはリサイズ後にシフトするか検討が必要）
            # ここでは元画像(512x288)でシフトを行い、その後にリサイズする
            # シフト量はすでに256px幅ベースで計算されているため、512px幅に換算して適用
            shift_px_scaled = shift_px * 2
            M = np.float32([[1, 0, shift_px_scaled], [0, 1, 0]])
            image = cv2.warpAffine(
                image, M, (image.shape[1], image.shape[0]),
                borderMode=cv2.BORDER_REPLICATE
            )
            # 補正は256px幅ベースで行う
            adjusted_angular_z -= shift_px * self.shift_vel_per_pixel

        # NEW Step: 環境光（明るさ・コントラスト）のランダム変動
        # 環境光の処理以外はシャノンサプライズブランチと同じにする
        if random.random() < 0.90:  # 90%の確率で適用
            alpha = random.uniform(0.5, 3.0)
            beta = random.randint(-30, 80)
            image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

        # Step4: リサイズ (512x288 -> 256x144)
        image = cv2.resize(image, (256, 144))

        return self._to_tensor(image), adjusted_angular_z

    def _to_tensor(self, image: np.ndarray) -> torch.Tensor:
        """
        画像を[0, 1]に正規化し、PyTorchのテンソル形式(C, H, W)に変換。
        """
        image_normalized = image.astype(np.float32) / 255.0
        return torch.from_numpy(image_normalized).permute(2, 0, 1)
