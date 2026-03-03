#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, Joy
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np
import time
import csv
from pathlib import Path
from typing import Optional, List, Tuple
from rclpy.qos import qos_profile_sensor_data

try:
    import pyzed.sl as sl
    ZED_SDK_AVAILABLE = True
except ImportError:
    ZED_SDK_AVAILABLE = False

# サンプリング間隔(秒)
SAMPLE_INTERVAL = 0.2
# 一定とする前進速度 (linear.x)
CONSTANT_LINEAR_X = 0.5 

class DataCollectionNode(Node):
    """
    データ収集用のROS 2ノード。
    カメラ画像とロボットの速度指令(cmd_vel)を同期して記録する。
    """
    def __init__(self) -> None:
        super().__init__('data_collection_node')

        self.declare_parameter('joy_button_toggle', 1)
        self.joy_button_toggle = self.get_parameter('joy_button_toggle').value

        self.bridge = CvBridge()
        
        # 最新のデータを保持する変数
        self.latest_image: Optional[Image] = None
        self.latest_angular_z: float = 0.0

        # 完成したデータ(画像, angular.z)のリスト
        self.collected_data: List[Tuple[np.ndarray, float]] = []
        self.last_sample_time: Optional[float] = None

        # データ収集のオン/オフ状態管理
        self.is_paused: bool = True
        self.last_joy_buttons: List[int] = []

        # ZED SDK用の変数
        self.zed_camera = None
        if not ZED_SDK_AVAILABLE:
            self.get_logger().error('ZED SDK not available. Install pyzed package.')
            raise RuntimeError('ZED SDK not available')
        self._initialize_zed_camera()

        # 速度指令(cmd_vel)データのサブスクリプションを追加
        self.create_subscription(Twist, '/cmd_vel', self.cmd_vel_callback, 10)

        # ジョイコントローラ
        self.create_subscription(Joy, '/joy', self.joy_callback, 10)
        
        self.create_timer(0.05, self.timer_callback)
        self.get_logger().info('⚪Create data started (Velocity Mode)')

    def _initialize_zed_camera(self) -> None:
        self.zed_camera = sl.Camera()
        init_params = sl.InitParameters()
        init_params.camera_resolution = sl.RESOLUTION.SVGA
        init_params.camera_fps = 30
        
        err = self.zed_camera.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            self.get_logger().error(f'Failed to open ZED camera: {err}')
            raise RuntimeError(f'Failed to open ZED camera: {err}')

        self.zed_image = sl.Mat()
        self.zed_runtime_params = sl.RuntimeParameters()
        self.get_logger().info('ZED camera initialized (SDK mode)')

    def _capture_data_from_zed(self) -> Optional[np.ndarray]:
        if self.zed_camera.grab(self.zed_runtime_params) != sl.ERROR_CODE.SUCCESS:
            return None
        self.zed_camera.retrieve_image(self.zed_image, sl.VIEW.LEFT)
        image = self.zed_image.get_data()
        height, width = image.shape[:2]
        return cv2.resize(image, (width // 2, height // 2))

    def cmd_vel_callback(self, msg: Twist) -> None:
        """ROSトピック経由で受信した速度指令から角速度を取得"""
        self.latest_angular_z = msg.angular.z

    def joy_callback(self, msg: Joy) -> None:
        current_buttons = msg.buttons
        
        # ボタンの状態変化を検出（押下のエッジ検出）
        if len(self.last_joy_buttons) == len(current_buttons):
            # トグルボタンが押された（OFF -> ON）
            if (len(current_buttons) > self.joy_button_toggle and 
                current_buttons[self.joy_button_toggle] == 1 and 
                self.last_joy_buttons[self.joy_button_toggle] == 0):
                
                self.is_paused = not self.is_paused
                if self.is_paused:
                    self.get_logger().info('⏸️ Data collection paused')
                else:
                    self.last_sample_time = None
                    self.get_logger().info('▶️ Data collection resumed')
        
        self.last_joy_buttons = list(current_buttons)

    def timer_callback(self) -> None:
        if self.is_paused:
            return

        current_time = time.time()
        
        image = self._capture_data_from_zed()
        if image is None: return

        # サンプリング間隔ごとにデータを保存
        if self.last_sample_time is None or current_time - self.last_sample_time >= SAMPLE_INTERVAL:
            self.collected_data.append((image, self.latest_angular_z))
            self.last_sample_time = current_time
            self.get_logger().info(f'🟡Collected data #{len(self.collected_data)} (angular_z: {self.latest_angular_z:.2f})')

    def save_data(self) -> None:
        if len(self.collected_data) == 0:
            self.get_logger().info('🔴No data to save')
            return

        package_root = Path(__file__).parent.parent
        data_base_dir = package_root / 'data'
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        dataset_dir = data_base_dir / f'{timestamp}_dataset'
        
        # 保存先ディレクトリ (マスクではなく生のRGB/BGR画像を想定)
        images_dir = dataset_dir / 'images'
        angular_vel_dir = dataset_dir / 'angular_vel'

        images_dir.mkdir(parents=True, exist_ok=True)
        angular_vel_dir.mkdir(parents=True, exist_ok=True)

        for idx, (image, angular_z) in enumerate(self.collected_data, start=1):
            image_path = images_dir / f'{idx:05d}.png'
            csv_path = angular_vel_dir / f'{idx:05d}.csv'

            cv2.imwrite(str(image_path), image)

            with open(str(csv_path), 'w', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                # ヘッダーと値を1行分だけ出力
                csv_writer.writerow(['linear.x', 'angular.z'])
                csv_writer.writerow([CONSTANT_LINEAR_X, angular_z])

        self.get_logger().info(f'🔵Saved {len(self.collected_data)} samples to {dataset_dir}')

def main(args=None) -> None:
    rclpy.init(args=args)
    node = DataCollectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Interrupted by user')
    finally:
        node.save_data()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
