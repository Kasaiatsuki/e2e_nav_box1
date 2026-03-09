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
        self.shift_pixels = [-12, -6, 0, 0, 6, 12]
        self.shift_vel_per_pixel = 0.15 / 6.0

    def preprocess_for_inference(self, image_bgr: np.ndarray) -> torch.Tensor:
        """
        推論前の画像処理。
        正規化、およびCHW形式への変換を行う。
        """
        return self._to_tensor(image_bgr)

    def augment_and_preprocess(self, image_bgr: np.ndarray, angular_z: float, idx: int) -> Tuple[torch.Tensor, float]:
        """
        学習用のデータ拡張と前処理。
        左右反転、水平シフト、正規化を行う。
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
            M = np.float32([[1, 0, shift_px], [0, 1, 0]])
            image = cv2.warpAffine(
                image, M, (image.shape[1], image.shape[0]),
                borderMode=cv2.BORDER_REPLICATE
            )
            # 右シフト(shift_px>0) → 廊下左寄り → 右に戻る(angular_z 減少)
            adjusted_angular_z -= shift_px * self.shift_vel_per_pixel

        return self._to_tensor(image), adjusted_angular_z

    def _to_tensor(self, image: np.ndarray) -> torch.Tensor:
        """
        画像を[0, 1]に正規化し、PyTorchのテンソル形式(C, H, W)に変換。
        """
        image_normalized = image.astype(np.float32) / 255.0
        return torch.from_numpy(image_normalized).permute(2, 0, 1)
