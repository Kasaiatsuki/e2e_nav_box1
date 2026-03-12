#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Empty
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np
import time
import csv
from pathlib import Path
from typing import Optional, List, Tuple
from rclpy.qos import qos_profile_sensor_data
from e2e_nav_box1.zed_capture import ZedCameraWrapper

# サンプリング間隔(秒)
SAMPLE_INTERVAL = 0.2

class DataCollectionNode(Node):
    """
    データ収集用のROS 2ノード。
    カメラ画像とロボットの速度指令(cmd_vel)を同期して記録する。
    """
    def __init__(self) -> None:
        super().__init__('data_collection_node')

        self.bridge = CvBridge()
        
        # 最新のデータを保持する変数
        self.latest_angular_z: float = 0.0

        # 完成したデータ(画像, angular.z)のリスト
        self.collected_data: List[Tuple[np.ndarray, float]] = []

        # データ収集のオン/オフ状態管理
        self.is_paused: bool = True

        # ZED SDK用の変数
        self.zed_camera = ZedCameraWrapper(fps=15)
        
        try:
            self.zed_camera.open()
            self.get_logger().info('ZED camera initialized (SDK mode)')
        except RuntimeError as e:
            self.get_logger().error(str(e))
            raise e

        # 速度指令(cmd_vel)データのサブスクリプションを追加
        self.create_subscription(Twist, '/cmd_vel', self.cmd_vel_callback, 10)

        # ジョイコントローラ
        self.create_subscription(Empty, 'flag', self._flag_callback, 10)
        #  タイマー自体をSAMPLE_INTERVALの周期で回し無駄なカメラアクセスを排除
        self.create_timer(SAMPLE_INTERVAL, self.timer_callback)
        self.get_logger().info('⚪Create data started (Velocity Mode)')
        # ZED API implementation was moved to ZedCameraWrapper

    def cmd_vel_callback(self, msg: Twist) -> None:
        """ROSトピック経由で受信した速度指令から角速度を取得"""
        self.latest_angular_z = msg.angular.z

    def _flag_callback(self, _msg: Empty) -> None:
        """/flagトピック受信時に録画をトグル(開始/停止)する"""
        self.is_paused = not self.is_paused
        
        if self.is_paused:
            self.get_logger().info('⏸️ Data collection PAUSED')
        else:
            self.get_logger().info('▶️ Data collection RECORDING')

    def timer_callback(self) -> None:
        if self.is_paused:
            return

        # ZEDカメラから画像を取得
        image = self.zed_camera.grab_image()
        if image is None:
            self.get_logger().warning('Failed to grab image from ZED camera')
            return

        # タイマー自体がSAMPLE_INTERVALの周期なので、時間判定を省略してそのまま保存
        self.collected_data.append((image, self.latest_angular_z))
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
                # 値のみを1行分出力
                csv_writer.writerow([angular_z])

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
        if hasattr(node, 'zed_camera') and node.zed_camera:
            node.zed_camera.close()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
