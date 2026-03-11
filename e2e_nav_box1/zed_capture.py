import pyzed.sl as sl
from typing import Optional
import numpy as np

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
        
        self.output_resolution = sl.Resolution(128, 72)

    def open(self) -> None:
        """カメラを開き、初期化する"""
        err = self.camera.open(self.init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"Failed to open ZED camera: {err}")

    def grab_image(self) -> Optional[np.ndarray]:
        """最新のカメラ画像(左目)をBGR形式(3ch)かつリサイズ済み(640x360)で取得して返す"""
        if self.camera.grab(self.runtime_params) == sl.ERROR_CODE.SUCCESS:
            self.camera.retrieve_image(self.zed_image, sl.VIEW.LEFT, sl.MEM.CPU, self.output_resolution)
            full_image = self.zed_image.get_data()
            # 4ch(BGRA) -> 3ch(BGR)
            return full_image[:, :, :3].copy()
        return None

    def close(self) -> None:
        """カメラを適切に閉じる"""
        self.camera.close()
