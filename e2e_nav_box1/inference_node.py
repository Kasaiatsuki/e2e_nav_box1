#!/usr/bin/env python3
"""
inference_node.py
役割: 学習済みのニューラルネットワークモデル(Behavioral Cloning)を利用して、カメラ画像からリアルタイムに角速度(angular.z)を推論し、ROS 2のTwistメッセージとして配信('/cmd_vel')するノードです。
自動運転システムの中で「運転手の脳」として機能し、入力画像からステアリング操作を直接決定します。
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Empty
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import torch
import numpy as np
import os
from pathlib import Path as FilePath
from rclpy.qos import qos_profile_system_default, qos_profile_sensor_data
from ament_index_python.packages import get_package_share_directory
from typing import Optional

import pyzed.sl as sl

# 一定とする前進速度 (linear.x)
CONSTANT_LINEAR_X = 0.8

class InferenceNode(Node):
    def __init__(self) -> None:
        super().__init__('inference_node')

        self.declare_parameter('model_name', 'e2e_model.pt')
        self.declare_parameter('interval_ms', 100)
        self.declare_parameter('debug_mode', True)
        self.declare_parameter('wait_for_flag', True)

        model_path = self.get_parameter('model_name').value
        interval_ms = self.get_parameter('interval_ms').value
        self.debug_mode_ = self.get_parameter('debug_mode').value
        wait_for_flag = self.get_parameter('wait_for_flag').value

        self.bridge = CvBridge()
        self.device = torch.device('cuda')
        self.zed_camera: Optional[sl.Camera] = None
        self.zed_image: Optional[sl.Mat] = None
        self.zed_runtime_params: Optional[sl.RuntimeParameters] = None

        package_share_directory = get_package_share_directory('e2e_nav_box1')
        weight_path = os.path.join(package_share_directory, 'weights', model_path)

        if os.path.exists(weight_path):
            self.model = torch.jit.load(weight_path, map_location=self.device)
            self.model.eval()
            self.get_logger().info(f'Model loaded from: {weight_path}')
        else:
            self.get_logger().warn(f'Model file not found: {weight_path}')
            self.model = None

        # ZED SDKは必須のため、そのまま初期化へ進む
        self._initialize_zed_camera()

        # 推論した速度をパブリッシュする先
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.last_angular_z = 0.0  # 前回の推論結果（カメラ失敗時に再利用）

        if self.debug_mode_:
            self.pub_debug_image = self.create_publisher(Image, 'e2e_planner/debug_image', qos_profile_system_default)
            self.get_logger().info('Debug mode enabled: publishing preprocessed images to e2e_planner/debug_image')

        self.timer = self.create_timer(interval_ms / 1000.0, self.timer_callback)

        # flagトピックを購読。wait_for_flag=True のときは起動直後にタイマーを停止して待機する
        self.create_subscription(Empty, 'flag', self._flag_callback, 10)
        if wait_for_flag:
            self.timer.cancel()
            self.get_logger().info('flagモード: "flag" トピック受信まで推論を待機中...')
        else:
            self.get_logger().info('通常モード: 推論を即時開始')

    def _flag_callback(self, _msg: Empty) -> None:
        """flagトピック受信時に推論タイマーを（再）起動する"""
        if not self.timer.is_ready() or self.timer.is_canceled():
            self.timer.reset()
            self.get_logger().info('フラグ受信: 推論開始')
        else:
            self.get_logger().info('フラグ受信: 推論はすでに動作中')

    def _initialize_zed_camera(self) -> None:
        self.zed_camera = sl.Camera()
        init_params = sl.InitParameters()
        init_params.camera_resolution = sl.RESOLUTION.HD720
        init_params.camera_fps = 30

        err = self.zed_camera.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            self.get_logger().error(f'Failed to open ZED camera: {err}')
            raise RuntimeError(f'Failed to open ZED camera: {err}')

        self.zed_image = sl.Mat()
        self.zed_runtime_params = sl.RuntimeParameters()
        self.get_logger().info('ZED camera initialized successfully')

    def _capture_image_from_zed(self) -> Optional[np.ndarray]:
        if self.zed_camera.grab(self.zed_runtime_params) == sl.ERROR_CODE.SUCCESS:
            self.zed_camera.retrieve_image(self.zed_image, sl.VIEW.LEFT)
            image = self.zed_image.get_data()
            return image
        return None

    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        bgr_image = image[:, :, :3] if image.shape[2] == 4 else image

        # 学習パイプライン(shift_px=0)と同じ変換:
        # 1280×720 全体を 640×360 にリサイズ（全幅視野を保持）
        # → 曲がり角の壁・通路も視野内に入る
        resized = cv2.resize(bgr_image, (640, 360), interpolation=cv2.INTER_LINEAR)

        # 正規化してPyTorchテンソルに変換
        image_normalized = resized.astype(np.float32) / 255.0
        tensor = torch.from_numpy(image_normalized).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self.device)

    def timer_callback(self) -> None:
        if self.model is None:
            return

        # --- 1. 画像の取得 ---
        cv_image = self._capture_image_from_zed()
        if cv_image is None:
            # カメラグラブ失敗時も前回の角速度で publish し続け、ypspur のタイムアウトを防ぐ
            twist_msg = Twist()
            twist_msg.linear.x = CONSTANT_LINEAR_X
            twist_msg.angular.z = self.last_angular_z
            self.cmd_vel_pub.publish(twist_msg)
            return
            
        from std_msgs.msg import Header
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = 'base_link'

        # --- 2. データの前処理 ---
        # OpenCVの画像を、AIが読めるPyTorchのテンソル形式（float32, 1x3x48x64など）に変換
        input_tensor = self.preprocess_image(cv_image)

        # デバッグ画像の配信
        if self.debug_mode_:
            # トリミングせずにそのままリサイズして配信
            bgr_image = cv_image[:, :, :3] if cv_image.shape[2] == 4 else cv_image
            resized_input = bgr_image
            debug_msg = self.bridge.cv2_to_imgmsg(resized_input, encoding='bgr8')
            debug_msg.header = header
            self.pub_debug_image.publish(debug_msg)

        # --- 3. AI推論 ---
        with torch.no_grad():
            output = self.model(input_tensor)

        # モデルの出力は [1, 1] (バッチサイズ1, 出力次元1) になっている前提
        # 出力された角速度を取得
        angular_z = float(output[0, 0].item())
        self.last_angular_z = angular_z  # 次回のカメラ失敗時のために保存

        # --- 4. Twistメッセージ(cmd_vel)の発行 ---
        twist_msg = Twist()
        twist_msg.linear.x = CONSTANT_LINEAR_X
        twist_msg.angular.z = angular_z
        self.cmd_vel_pub.publish(twist_msg)

def main(args=None) -> None:
    rclpy.init(args=args)
    node = InferenceNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
