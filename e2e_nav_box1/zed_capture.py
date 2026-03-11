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
        """最新のカメラ画像(左目)をBGR形式(3ch)かつリサイズ済み(640x360)で取得して返す"""
        if self.camera.grab(self.runtime_params) == sl.ERROR_CODE.SUCCESS:
            # ネイティブ解像度(例: HD720=1280x720)で画像を取得する。
            # ※ここで output_resolution を渡すとZED SDK内部リサイズによりstride(メモリパディング)が崩れて画像が裂ける
            self.camera.retrieve_image(self.zed_image, sl.VIEW.LEFT, sl.MEM.CPU)
            full_image = self.zed_image.get_data()
            
            # ZEDはデフォルトで4ch(BGRA)を返す。メモリを連続化するためにcvtColorでBGRへ。
            if full_image is not None and full_image.shape[2] == 4:
                bgr_image = cv2.cvtColor(full_image, cv2.COLOR_BGRA2BGR)
            else:
                bgr_image = full_image
                
            # OpenCVで安全に 640x360 にリサイズ
            return cv2.resize(bgr_image, (self.output_resolution.width, self.output_resolution.height), interpolation=cv2.INTER_LINEAR)
        return None

    def close(self) -> None:
        """カメラを適切に閉じる"""
        self.camera.close()
