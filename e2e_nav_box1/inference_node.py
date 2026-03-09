#!/usr/bin/env python3
"""
inference_node.py
役割: 学習済みのニューラルネットワークモデル(Behavioral Cloning)を利用して、カメラ画像からリアルタイムに角速度(angular.z)を推論し、ROS 2のTwistメッセージとして配信('/cmd_vel')するノードです。
自動運転システムの中で「運転手の脳」として機能し、入力画像からステアリング操作を直接決定します。
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import torch
import numpy as np
import os
from rclpy.qos import qos_profile_system_default
from ament_index_python.packages import get_package_share_directory
from typing import Optional

from e2e_nav_box1.zed_capture import ZedCameraWrapper
from e2e_nav_box1.image_processor import ImageProcessor



class InferenceNode(Node):
    def __init__(self) -> None:
        super().__init__('inference_node')

        self.declare_parameter('model_name', rclpy.Parameter.Type.STRING)
        self.declare_parameter('interval_ms', rclpy.Parameter.Type.INTEGER)
        self.declare_parameter('debug_mode', rclpy.Parameter.Type.BOOL)
        self.declare_parameter('wait_for_flag', rclpy.Parameter.Type.BOOL)
        self.declare_parameter('linear_x', rclpy.Parameter.Type.DOUBLE)

        model_path = self.get_parameter('model_name').value
        interval_ms = self.get_parameter('interval_ms').value
        self.debug_mode_ = self.get_parameter('debug_mode').value
        self.is_autonomous = not self.get_parameter('wait_for_flag').value
        self.linear_x = self.get_parameter('linear_x').value

        self.bridge = CvBridge()
        self.device = torch.device('cuda')
        self.processor = ImageProcessor()
        self.zed_camera = ZedCameraWrapper(fps=30)
        try:
            self.zed_camera.open()
            self.get_logger().info('ZED camera initialized successfully')
        except RuntimeError as e:
            self.get_logger().error(str(e))
            raise e

        package_share_directory = get_package_share_directory('e2e_nav_box1')
        weight_path = os.path.join(package_share_directory, 'weights', model_path)
        self.model = torch.jit.load(weight_path, map_location=self.device)
        self.model.eval()
        self.get_logger().info(f'Model loaded from: {weight_path}')

        # 推論した速度をパブリッシュする先
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        if self.debug_mode_:
            self.pub_debug_image = self.create_publisher(Image, 'e2e_planner/debug_image', qos_profile_system_default)
            self.get_logger().info('Debug mode enabled: publishing preprocessed images to e2e_planner/debug_image')

        self.timer = self.create_timer(interval_ms / 1000.0, self.timer_callback)

        # autonomousトピック(Bool)を購読して、走行状態を切り替える
        self.create_subscription(Bool, 'autonomous', self._autonomous_callback, 10)
        self.get_logger().info('Inference node started (Waiting for autonomous=True)')

    def _autonomous_callback(self, msg: Bool) -> None:
        """autonomousトピック受信時に走行状態を更新する"""
        self.is_autonomous = msg.data
        status = "ENABLED" if self.is_autonomous else "DISABLED"
        self.get_logger().info(f'Autonomous mode: {status}')

        # ZED API implementation was moved to ZedCameraWrapper

    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        ImageProcessorを使用して、AI推論用の前処理を行う。
        """
        tensor = self.processor.preprocess_for_inference(image)
        # バッチ次元の追加とデバイス転送
        return tensor.unsqueeze(0).to(self.device)

    def timer_callback(self) -> None:
        if not self.is_autonomous:
            return

        # --- 1. 画像の取得 ---
        cv_image = self.zed_camera.grab_image()
        if cv_image is None:
            return
        # --- 2. データの前処理 ---
        # OpenCVの画像を、AIが読めるPyTorchのテンソル形式（float32, 1x3x48x64など）に変換
        input_tensor = self.preprocess_image(cv_image)

        # デバッグ画像の配信
        if self.debug_mode_:
            # トリミングせずにそのままリサイズして配信
            resized_input = cv_image
            debug_msg = self.bridge.cv2_to_imgmsg(resized_input, encoding='bgr8')
            debug_msg.header.stamp = self.get_clock().now().to_msg()
            debug_msg.header.frame_id = 'base_link'
            self.pub_debug_image.publish(debug_msg)

        # --- 3. AI推論 ---
        with torch.no_grad():
            output = self.model(input_tensor)

        # モデルの出力は [1, 1] (バッチサイズ1, 出力次元1) になっている前提
        # 出力された角速度を取得
        angular_z = float(output[0, 0].item())

        # --- 4. Twistメッセージ(cmd_vel)の発行 ---
        twist_msg = Twist()
        twist_msg.linear.x = self.linear_x
        twist_msg.angular.z = angular_z
        self.cmd_vel_pub.publish(twist_msg)

def main(args=None) -> None:
    rclpy.init(args=args)
    node = InferenceNode()
    rclpy.spin(node)
    if hasattr(node, 'zed_camera') and node.zed_camera:
        node.zed_camera.close()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
