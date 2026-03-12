import pyzed.sl as sl
from typing import Optional
import numpy as np
import cv2

class ZedCameraWrapper:
    """
    ZEDカメラの初期化と画像取得をカプセル化したラッパークラス。
    データ収集(create_data.py)および推論(inference_node.py)で再利用される。
    """
    def __init__(self, fps: int = 15, resolution=sl.RESOLUTION.HD720) -> None:
        self.fps = fps
        self.resolution = resolution
        
        self.camera = sl.Camera()
        self.init_params = sl.InitParameters()
        self.init_params.camera_resolution = self.resolution
        self.init_params.camera_fps = self.fps
        
        self.zed_image = sl.Mat()
        self.runtime_params = sl.RuntimeParameters()
        
        self.output_size = (128, 72)
        self.output_resolution = sl.Resolution(640, 360)

    def open(self) -> None:
        """カメラを開き、初期化する"""
        err = self.camera.open(self.init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"Failed to open ZED camera: {err}")

    def grab_image(self) -> Optional[np.ndarray]:
        """最新のカメラ画像(左目)をBGR形式(3ch)かつリサイズ済み(128x72)で取得して返す"""
        if self.camera.grab(self.runtime_params) == sl.ERROR_CODE.SUCCESS:
            # 3月7日の正常動作時の実装をベースに、リサイズ先を 128x72 に変更:
            # 1. ネイティブ解像度で取得 (SDK内部リサイズを避ける)
            # 2. NumPyスライスで3チャンネル抽出
            # 3. OpenCVで 128x72 にリサイズ
            self.camera.retrieve_image(self.zed_image, sl.VIEW.LEFT)
            image = self.zed_image.get_data()
            if image is None: return None
            
            # 4ch(RGBA) -> 3ch(BGR)
            bgr_image = image[:, :, :3] if image.shape[2] == 4 else image
            
            # 128x72 にリサイズ
            return cv2.resize(bgr_image, (128, 72), interpolation=cv2.INTER_LINEAR)
        return None

    def close(self) -> None:
        """カメラを適切に閉じる"""
        self.camera.close()
