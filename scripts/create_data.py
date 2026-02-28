#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, Joy, PointCloud2
from geometry_msgs.msg import PoseWithCovarianceStamped
from cv_bridge import CvBridge
import cv2
import numpy as np
import os
import time
import csv
import copy
from pathlib import Path
from collections import deque
from typing import Optional, List, Tuple, Deque
from tf_transformations import euler_from_quaternion
from rclpy.qos import qos_profile_sensor_data
import sensor_msgs_py.point_cloud2 as pc2
import pymap3d as pm
try:
    import pyzed.sl as sl
    ZED_SDK_AVAILABLE = True
except ImportError:
    ZED_SDK_AVAILABLE = False

# サンプリング間隔(秒)とウェイポイントの間隔(秒)、取得するウェイポイントの数
SAMPLE_INTERVAL = 0.2
WAYPOINT_INTERVAL = 0.5
NUM_WAYPOINTS = 10

class Sample:
    """
    データセットの1サンプルを管理するクラス。
    画像、タイムスタンプ、基準点となる姿勢、点群データなどを保持する。
    """
    def __init__(self, image: np.ndarray, timestamp: float, reference_pose: PoseWithCovarianceStamped, point_cloud: Optional[np.ndarray] = None):
        self.image: np.ndarray = image
        self.timestamp: float = timestamp
        self.reference_pose: PoseWithCovarianceStamped = reference_pose
        self.point_cloud: Optional[np.ndarray] = point_cloud
        
        # 将来のウェイポイント(軌跡)のリスト
        self.waypoints: List[Tuple[float, float]] = []
        # 各ウェイポイントを取得すべき目標時刻のリスト
        self.target_times: List[float] = [timestamp + WAYPOINT_INTERVAL * (i + 1) for i in range(NUM_WAYPOINTS)]

class DataCollectionNode(Node):
    """
    データ収集用のROS 2ノード。
    カメラ画像、点群、自己位置推定(Pose)を同期して記録し、データセットを作成する。
    """
    def __init__(self) -> None:
        super().__init__('data_collection_node')

        # ZED SDKを直接使用するかどうかのフラグ
        self.declare_parameter('sdk_flag', False)
        self.sdk_flag_ = self.get_parameter('sdk_flag').value

        self.bridge: CvBridge = CvBridge()
        
        # ROSトピック経由で取得した最新データを保持する変数
        self.latest_image: Optional[Image] = None
        self.latest_pose: Optional[PoseWithCovarianceStamped] = None
        self.latest_pointcloud: Optional[PointCloud2] = None

        # 収集プロセス中のデータ保持用
        self.samples: List[Sample] = []
        self.pose_history: Deque[Tuple[float, PoseWithCovarianceStamped]] = deque()
        # 完成したデータ(画像, ウェイポイント, 点群)のリスト
        self.collected_data: List[Tuple[np.ndarray, List[Tuple[float, float]], Optional[np.ndarray]]] = []
        self.last_sample_time: Optional[float] = None

        # データ収集のオン/オフ状態管理
        self.is_paused: bool = True
        self.prev_button_state: int = 0

        # ZED SDK用の変数
        self.zed_camera: Optional[sl.Camera] = None
        self.zed_image: Optional[sl.Mat] = None
        self.zed_point_cloud: Optional[sl.Mat] = None
        self.zed_pose: Optional[sl.Pose] = None
        self.zed_runtime_params: Optional[sl.RuntimeParameters] = None

        if self.sdk_flag_:
            # ZED SDKを使用する場合の初期化
            if not ZED_SDK_AVAILABLE:
                self.get_logger().error('ZED SDK not available. Install pyzed package.')
                raise RuntimeError('ZED SDK not available')
            self._initialize_zed_camera()
        else:
            # ROS 2トピックをサブスクライブして画像と点群を取得する場合の初期化
            self.create_subscription(Image, '/zed/zed_node/rgb/image_rect_color', self.image_callback, qos_profile_sensor_data)
            self.create_subscription(PointCloud2, '/zed/zed_node/pointcloud', self.pointcloud_callback, qos_profile_sensor_data)

        # 自己位置推定(Pose)データのサブスクリプション (VectorNav等から)
        self.create_subscription(PoseWithCovarianceStamped, '/vectornav/pose', self.pose_callback, qos_profile_sensor_data)

        # ジョイコントローラによるデータ収集の開始/一時停止制御
        self.create_subscription(Joy, '/joy', self.joy_callback, 10)
        
        # 0.1秒周期のメインループ用タイマー
        self.create_timer(0.1, self.timer_callback)

        self.get_logger().info('⚪Create data started')

    def _initialize_zed_camera(self) -> None:
        """ZED SDK経由でZEDカメラを初期化する"""
        self.zed_camera = sl.Camera()
        init_params = sl.InitParameters()
        init_params.camera_resolution = sl.RESOLUTION.SVGA
        init_params.camera_fps = 30
        init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE
        init_params.coordinate_units = sl.UNIT.METER

        err = self.zed_camera.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            self.get_logger().error(f'Failed to open ZED camera: {err}')
            raise RuntimeError(f'Failed to open ZED camera: {err}')

        # Positional Tracking(自己位置推定)を有効化
        tracking_params = sl.PositionalTrackingParameters()
        err = self.zed_camera.enable_positional_tracking(tracking_params)
        if err != sl.ERROR_CODE.SUCCESS:
            self.get_logger().error(f'Failed to enable positional tracking: {err}')
            raise RuntimeError(f'Failed to enable positional tracking: {err}')

        self.zed_image = sl.Mat()
        self.zed_point_cloud = sl.Mat()
        self.zed_pose = sl.Pose()
        self.zed_runtime_params = sl.RuntimeParameters()
        self.get_logger().info('ZED camera initialized with tracking and depth sensing')

    def _capture_data_from_zed(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """ZED SDKから最新の画像と点群データを取得し、フォーマット変換する"""
        if self.zed_camera.grab(self.zed_runtime_params) != sl.ERROR_CODE.SUCCESS:
            return None, None

        # 画像の取得とリサイズ (幅/高さ を半分に)
        self.zed_camera.retrieve_image(self.zed_image, sl.VIEW.LEFT)
        image = self.zed_image.get_data()
        height, width = image.shape[:2]
        resized_image = cv2.resize(image, (width // 2, height // 2))

        # 点群の取得
        self.zed_camera.retrieve_measure(self.zed_point_cloud, sl.MEASURE.XYZRGBA)
        point_cloud = self.zed_point_cloud.get_data()

        return resized_image, point_cloud

    def image_callback(self, msg: Image) -> None:
        """ROSトピック経由で受信した画像を保持"""
        self.latest_image = msg

    def pose_callback(self, msg: PoseWithCovarianceStamped) -> None:
        """ROSトピック経由で受信したPoseを保持"""
        self.latest_pose = msg

    def pointcloud_callback(self, msg: PointCloud2) -> None:
        """ROSトピック経由で受信した点群データを保持"""
        self.latest_pointcloud = msg

    def _convert_pointcloud2_to_array(self, pointcloud_msg: PointCloud2) -> np.ndarray:
        """ROSのPointCloud2メッセージをnumpy配列(x, y, z, rgb)に変換"""
        points_list = [[point[0], point[1], point[2], point[3]]
                       for point in pc2.read_points(pointcloud_msg, skip_nans=True, field_names=("x", "y", "z", "rgb"))]
        return np.array(points_list, dtype=np.float32)

    def joy_callback(self, msg: Joy) -> None:
        """ジョイスティック入力コールバック。ボタン[2]でデータ収集の再生/一時停止を切り替える"""
        if len(msg.buttons) > 2:
            current_button_state = msg.buttons[2]
            # ボタンが押された瞬間(0->1)にトグル処理
            if current_button_state == 1 and self.prev_button_state == 0:
                self.is_paused = not self.is_paused
                if self.is_paused:
                    self.get_logger().info('⏸️ Data collection paused')
                else:
                    # 収集再開時には履歴をクリアして新鮮な状態でスタート
                    self.pose_history.clear()
                    self.samples.clear()
                    self.last_sample_time = None
                    self.get_logger().info('▶️ Data collection resumed')
            self.prev_button_state = current_button_state

    def timer_callback(self) -> None:
        """定期的に呼び出されるメインの処理。データのサンプリングとウェイポイント収集を行う"""
        if self.is_paused:
            return

        current_time = time.time()

        # 常に最新のPose(位置・姿勢)を履歴として保存
        if self.latest_pose is not None:
            self.pose_history.append((current_time, copy.deepcopy(self.latest_pose)))

        if self.sdk_flag_:
            # ZED SDKからデータ取得するモードの場合
            image, point_cloud = self._capture_data_from_zed()
            if image is None or self.latest_pose is None:
                return

            # サンプリング間隔(SAMPLE_INTERVAL)に達したら新しいSampleを作成
            if self.last_sample_time is None or current_time - self.last_sample_time >= SAMPLE_INTERVAL:
                sample = Sample(image, current_time, copy.deepcopy(self.latest_pose), point_cloud)
                self.samples.append(sample)
                self.last_sample_time = current_time
        else:
            # ROSトピックからデータ取得するモードの場合
            if self.latest_image is not None and self.latest_pose is not None:
                # サンプリング間隔(SAMPLE_INTERVAL)に達したら新しいSampleを作成
                if self.last_sample_time is None or current_time - self.last_sample_time >= SAMPLE_INTERVAL:
                    cv_image = self.bridge.imgmsg_to_cv2(self.latest_image, desired_encoding='bgra8')
                    point_cloud = self._convert_pointcloud2_to_array(self.latest_pointcloud) if self.latest_pointcloud is not None else None
                    sample = Sample(cv_image, current_time, copy.deepcopy(self.latest_pose), point_cloud)
                    self.samples.append(sample)
                    self.last_sample_time = current_time

        # 収集途中の各Sampleについて、将来の対応する時刻のウェイポイントを取得・追加
        for sample in self.samples:
            self.collect_waypoints_for_sample(sample)

        # 指定数のウェイポイント収集が完了したSampleを完成済みデータとして保存
        completed_samples = [sample for sample in self.samples if len(sample.waypoints) == NUM_WAYPOINTS]
        for sample in completed_samples:
            self.collected_data.append((sample.image, sample.waypoints, sample.point_cloud))
            self.get_logger().info(f'🟡Collected data #{len(self.collected_data)}')

        # 未完成のSampleのみ後段に残す
        self.samples = [sample for sample in self.samples if len(sample.waypoints) < NUM_WAYPOINTS]

        # 役割を終えた過去のPose履歴を削除(メモリ節約)
        self.cleanup_pose_history()

    def collect_waypoints_for_sample(self, sample: Sample) -> None:
        """
        対象Sampleの不足しているウェイポイントについて、
        目標時刻(target_time)に達していればPose履歴から取得し、相対座標を計算して追加する
        """
        for i in range(len(sample.waypoints), NUM_WAYPOINTS):
            target_time = sample.target_times[i]
            pose = self.find_closest_pose(target_time)
            if pose is not None:
                # 基準姿勢(Sample作成時)から見た対象時刻での相対的なロボット座標(x, y)を算出
                x, y = self.transform_to_robot_frame(sample.reference_pose, pose)
                sample.waypoints.append((x, y))
            else:
                # まだ未来の姿勢データが得られていない場合はループ脱出
                break

    def find_closest_pose(self, target_time: float) -> Optional[PoseWithCovarianceStamped]:
        """指定された目標時刻(target_time)以降の最初のPoseを履歴から探す"""
        for t, pose in self.pose_history:
            if t >= target_time:
                return pose
        return None

    def cleanup_pose_history(self) -> None:
        """
        最も古い「未取得ウェイポイントの目標時刻」よりも前のPose履歴は不要になるため削除する
        """
        if not self.samples or not self.pose_history:
            return
        incomplete_samples = [s for s in self.samples if len(s.waypoints) < NUM_WAYPOINTS]
        if not incomplete_samples:
            return
            
        # 必要な最も古いPoseの時刻を取得
        min_target_time = min(sample.target_times[len(sample.waypoints)] for sample in incomplete_samples)
        
        # それより古い履歴をポップする
        while self.pose_history and self.pose_history[0][0] < min_target_time:
            self.pose_history.popleft()

    def transform_to_robot_frame(self, reference_pose: PoseWithCovarianceStamped, current_pose: PoseWithCovarianceStamped) -> Tuple[float, float]:
        """
        グローバル地図座標(ECEF等)上の2つのPoseを受け取り、
        reference_pose(基準姿勢)から見た current_pose の相対座標(x, y)に変換する。
        """
        # 基準位置のECEF座標 -> Geodetic(緯度/経度/高度)
        x0_ecef = reference_pose.pose.pose.position.x
        y0_ecef = reference_pose.pose.pose.position.y
        z0_ecef = reference_pose.pose.pose.position.z
        lat0, lon0, alt0 = pm.ecef2geodetic(x0_ecef, y0_ecef, z0_ecef)

        # 基準の向き(クォータニオン) -> ヨー角(Yaw)
        q0 = reference_pose.pose.pose.orientation
        _, _, yaw0 = euler_from_quaternion([q0.x, q0.y, q0.z, q0.w])

        # 現在位置のECEF座標 -> 基準位置から見たENU(East, North, Up)座標系へ
        xi_ecef = current_pose.pose.pose.position.x
        yi_ecef = current_pose.pose.pose.position.y
        zi_ecef = current_pose.pose.pose.position.z
        e, n, u = pm.ecef2enu(xi_ecef, yi_ecef, zi_ecef, lat0, lon0, alt0)

        # ENU座標をロボットの向いている方向(Yaw)に合わせて回転 (前方=X, 左方=Y)
        x_robot = -e * np.sin(yaw0) + n * np.cos(yaw0)
        y_robot = -e * np.cos(yaw0) - n * np.sin(yaw0)

        return x_robot, y_robot

    def save_data(self) -> None:
        """
        完成したデータセットをファイルに保存する。
        画像(PNG)、ウェイポイント軌跡(CSV)、点群(NPY)を出力する。
        """
        if len(self.collected_data) == 0:
            self.get_logger().info('🔴No data to save')
            return

        # パッケージの 'data' フォルダ内にタイムスタンプのディレクトリを作成
        package_root = Path(__file__).parent.parent
        data_base_dir = package_root / 'data'
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        dataset_dir = data_base_dir / f'{timestamp}_dataset'
        
        images_dir = dataset_dir / 'images'
        path_dir = dataset_dir / 'path'
        pointclouds_dir = dataset_dir / 'pointclouds'

        images_dir.mkdir(parents=True, exist_ok=True)
        path_dir.mkdir(parents=True, exist_ok=True)
        pointclouds_dir.mkdir(parents=True, exist_ok=True)

        # 連番でファイル保存
        for idx, (image, waypoints, point_cloud) in enumerate(self.collected_data, start=1):
            image_path = images_dir / f'{idx:05d}.png'
            waypoints_path = path_dir / f'{idx:05d}.csv'
            pointcloud_path = pointclouds_dir / f'{idx:05d}.npy'

            # 画像出力
            cv2.imwrite(str(image_path), image)

            # ウェイポイント出力(CSV形式: x, y)
            with open(str(waypoints_path), 'w', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(['x', 'y'])
                for x, y in waypoints:
                    csv_writer.writerow([x, y])

            # 点群出力(存在する場合)
            if point_cloud is not None:
                np.save(str(pointcloud_path), point_cloud)

        self.get_logger().info(f'🔵Saved {len(self.collected_data)} samples to {dataset_dir}')

def main(args=None) -> None:
    """メイン関数。ROS 2ノードを起動しスピンさせる"""
    rclpy.init(args=args)
    node = DataCollectionNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Interrupted by user')
    finally:
        # ノード終了時に保存処理を実行
        node.save_data()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
