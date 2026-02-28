"""
inference_node.py
役割: 学習済みのニューラルネットワークモデルを利用して、カメラ画像からリアルタイムに将来の走行軌道（Waypoint）を推論（予測）し、ROS 2のPathメッセージとして配信（パブリッシュ）するノードです。
自動運転システムの中で「運転手の脳」として機能し、入力画像から「次にどう走るべきか」を決定します。
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, Pose, Point
from cv_bridge import CvBridge
import cv2
import torch
import numpy as np
import os
from pathlib import Path as FilePath
from rclpy.qos import qos_profile_system_default, qos_profile_sensor_data
from ament_index_python.packages import get_package_share_directory
from scipy.interpolate import splprep, splev
from typing import Optional, Tuple

try:
    import pyzed.sl as sl
    ZED_SDK_AVAILABLE = True
except ImportError:
    ZED_SDK_AVAILABLE = False

def denormalize_waypoints(normalized: np.ndarray) -> np.ndarray:
    """
    モデルが出力した正規化された範囲（-1.0〜1.0）のWaypoint座標を、
    現実のメートル単位（x: 0〜10m, y: -3〜3m程度）に復元（逆正規化）します。
    train.pyで行った正規化の全く逆の計算を行っています。
    """
    denormalized = normalized.copy()
    denormalized[0::2] = (normalized[0::2] + 1.0) * 5.0
    denormalized[1::2] = (normalized[1::2] + 1.0) * 3.0 - 3.0
    return denormalized

class InferenceNode(Node):
    def __init__(self) -> None:
        super().__init__('inference_node')

        # パラメータサーバーから起動時の設定値（引数）を取得するための宣言
        # model_name: 読み込むE2Eモデルのファイル名（重みデータ）
        # interval_ms: AI推論ループを自動実行する周期(ミリ秒)
        # sdk_flag: 高速なZED SDKを直接使用するか(True)、ROSの画像トピックを間接的に購読するか(False)
        # debug_mode: 推論時、AIがどういうマスク画像を見ているかを確認するための画像を配信するか
        self.declare_parameter('model_name', 'model.pt')
        self.declare_parameter('interval_ms', 100)
        self.declare_parameter('sdk_flag', True)
        self.declare_parameter('debug_mode', True)

        # 宣言したパラメータを変数として取り出す
        model_path = self.get_parameter('model_name').value
        interval_ms = self.get_parameter('interval_ms').value
        self.sdk_flag_ = self.get_parameter('sdk_flag').value
        self.debug_mode_ = self.get_parameter('debug_mode').value

        # ROS(Image)とOpenCV(cv2)の画像形式を相互に変換するためのブリッジツール
        self.bridge = CvBridge()
        # AI推論を高速なGPU(CUDA)で実行することを指定
        self.device = torch.device('cuda')
        self.latest_image = None
        self.zed_camera: Optional[sl.Camera] = None
        self.zed_image: Optional[sl.Mat] = None
        self.zed_runtime_params: Optional[sl.RuntimeParameters] = None

        # ROS 2環境下でパッケージがインストールされた共有ディレクトリ(share)の絶対パスを取得し、
        # E2Eモデル(train.pyで作ったもの)の場所を特定する
        package_share_directory = get_package_share_directory('e2e_nav_box1')
        weight_path = os.path.join(package_share_directory, 'weights', model_path)

        # E2E Plannerモデルの読み込み
        if os.path.exists(weight_path):
            # map_location=self.device により、モデルをGPUメモリ上に直接ロードする
            self.model = torch.jit.load(weight_path, map_location=self.device)
            # train.pyの時とは違い、AIを推論専用のモード(学習用機能をストップ)へ切り替え
            self.model.eval()
        else:
            self.get_logger().warn(f'Model file not found: {weight_path}')
            self.model = None

        # ROS 2トピックのサブスクライバー（購読者）の設定
        # ZED SDKを使わない場合、カメラからの画像トピックを受け取って最新画像を確保し続ける
        self.sub = self.create_subscription(Image, '/zed/zed_node/left/image_rect_color', self.image_callback, qos_profile_sensor_data)
        if self.sdk_flag_:
            if not ZED_SDK_AVAILABLE:
                self.get_logger().error('ZED SDK not available. Install pyzed package.')
                raise RuntimeError('ZED SDK not available')
            self._initialize_zed_camera()
        else:
            self.sub = self.create_subscription(Image, '/zed/zed_node/rgb/image_rect_color', self.image_callback, qos_profile_sensor_data)

        # AIの推論結果である経路(Path)を、後段の制御システム(Pure Pursuitなど)へパブリッシュ（配信）するための設定
        # path_raw: AIが出力した10個の点をそのまま直線で結んだカクカクの経路
        self.pub_raw = self.create_publisher(Path, 'e2e_planner/path_raw', qos_profile_system_default)
        # path: B-スプライン曲線という技術で滑らかに補間処理された、自動運転制御用の綺麗な経路
        self.pub = self.create_publisher(Path, 'e2e_planner/path', qos_profile_system_default)

        # デバッグモードの場合、AIが処理している真っ最中の「赤いフィルタ画像」などを人間が見れるように配信する
        if self.debug_mode_:
            self.pub_debug_image = self.create_publisher(Image, 'e2e_planner/debug_image', qos_profile_system_default)
            self.get_logger().info('Debug mode enabled: publishing preprocessed images to e2e_planner/debug_image')

        # interval_ms (例: 100ms = 0.1秒) ごとに timer_callback 関数を自動的に呼び出すタイマー中枢機能
        self.timer = self.create_timer(interval_ms / 1000.0, self.timer_callback)

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
        self.get_logger().info('ZED camera initialized successfully')

    def _capture_image_from_zed(self) -> Optional[np.ndarray]:
        if self.zed_camera.grab(self.zed_runtime_params) == sl.ERROR_CODE.SUCCESS:
            self.zed_camera.retrieve_image(self.zed_image, sl.VIEW.LEFT)
            image = self.zed_image.get_data()
            # Resize image to half size
            height, width = image.shape[:2]
            resized_image = cv2.resize(image, (width // 2, height // 2))
            return resized_image
        return None

    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        カメラ画像を受け取り、特定範囲をクロップ・リサイズして、
        AIモデルに入力するためのPyTorchのテンソル（[1, 3, 48, 64]の形）に変換します。
        
        高速化のため、ZEDやOpenCVのデフォルトフォーマットであるBGR/BGRAから
        RGBへの変換を行わず、そのまま（Alphaチャンネルがあれば除去してBGRのまま）処理します。
        """
        # BGRA(4チャンネル)の場合は、余分なAlphaだけ削って3チャンネル(BGR)にする
        bgr_image = image[:, :, :3] if image.shape[2] == 4 else image

        # ネットワークに入力する関心領域(ROI)のみをクロップ(x:40〜440)
        cropped_image = bgr_image[:, 40:440]

        # 学習時と同じサイズ(64x48)へリサイズ(バイリニア補間)
        resized_image = cv2.resize(cropped_image, (64, 48), interpolation=cv2.INTER_LINEAR)

        # PyTorchのfloat32テンソルに変換し、[0, 1]スケールへ正規化
        image_normalized = resized_image.astype(np.float32) / 255.0

        # 次元をHWCからCHWに変更し、バッチ次元(unsqueeze)を追加
        tensor = torch.from_numpy(image_normalized).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self.device)

    def image_callback(self, msg: Image) -> None:
        self.latest_image = msg

    def timer_callback(self) -> None:
        """
        タイマーによって定期的に（例：0.1秒おきに）呼び出され、
        「画像取得 → 前処理 → AI推論 → 後処理 → パス送信」という自動運転の一連のサイクルを実行するメイン関数
        """
        # --- 1. 画像の取得 ---
        if self.sdk_flag_:
            # ZED SDKから直接画像を取得し、タイムスタンプ等を構築
            cv_image = self._capture_image_from_zed()
            if cv_image is None:
                return
            from std_msgs.msg import Header
            header = Header()
            header.stamp = self.get_clock().now().to_msg()
            header.frame_id = 'base_link' # 車両の中心座標系を意味する
        else:
            # ROS 2トピックで受信した最新画像をOpenCV形式(cv2)に変換
            if self.latest_image is None:
                return
            cv_image = self.bridge.imgmsg_to_cv2(self.latest_image, desired_encoding='bgra8')
            header = self.latest_image.header

        # --- 2. データの前処理 ---
        # OpenCVの画像を、AIが読めるPyTorchのテンソル形式（float32, 1x3x48x64など）に変換
        input_tensor = self.preprocess_image(cv_image)

        # --- デバッグ表示用（AIに入力されるBGR画像を可視化） ---
        if self.debug_mode_:
            # RGB変換せずに、入力のままの画像（アルファ抜き等の処理用）をトリミング
            bgr_image = cv_image[:, :, :3] if cv_image.shape[2] == 4 else cv_image
            cropped_image = bgr_image[:, 40:440]
            resized_input = cv2.resize(cropped_image, (64, 48))
            debug_msg = self.bridge.cv2_to_imgmsg(resized_input, encoding='bgr8')
            debug_msg.header = header
            self.pub_debug_image.publish(debug_msg)

        # AIモデルによる未来の軌道（Waypoint）の推論実行。学習時と同様にno_grad()で無駄な計算メモリを節約します。
        with torch.no_grad():
            output = self.model(input_tensor)

        # 推論結果（-1.0〜1.0）を1次元配列にし、現実のメートル単位の座標に逆正規化
        output_normalized = output.cpu().numpy().flatten()
        output_denormalized = denormalize_waypoints(output_normalized)
        output_denormalized_tensor = torch.from_numpy(output_denormalized).unsqueeze(0)

        # 点をそのまま結んだカクカクの生パス（Raw）を作成してパブリッシュ
        path_raw_msg = self.create_path_from_output(output_denormalized_tensor, header)
        self.pub_raw.publish(path_raw_msg)

        # ハンドル操作がスムーズになるように、点と点の間をB-スプライン曲線で滑らかに補間したパスを作成してパブリッシュ
        path_smooth_msg = self.apply_bspline_smoothing(output_denormalized_tensor, header)
        self.pub.publish(path_smooth_msg)

    def apply_bspline_smoothing(self, output: torch.Tensor, header) -> Path:
        """
        AIが予測したばらばらの点（ウェイポイント）を、自然で滑らかな一本の曲線データに変換します。
        これを行わないと、車が点に向かってカクカクと不自然なハンドリングをしてしまいます。
        """
        # [x, y, x, y...] の1次元テンソルを、[[x1,y1], [x2,y2]...] の2次元配列に再構築
        waypoints = output.cpu().numpy().reshape(-1, 2)
        x = waypoints[:, 0]
        y = waypoints[:, 1]

        # SciPyライブラリを使ったB-Spline曲線の作成
        # s=0.1: smoothing factor（値が大きいほど元の点を無視して滑らかなカーブを描く。0だと元の点を必ず通過する）
        # k=3: スプラインの次数（3次曲線=S字のようなカーブを描ける標準的な滑らかさ）
        tck, u = splprep([x, y], s=0.1, k=3)
        # 点と点の間に、均等に30個の細かい新しい点を刻み直す
        u_new = np.linspace(0, 1, 30)
        x_smooth, y_smooth = splev(u_new, tck)

        # ROS 2のPath（経路）メッセージに組み立てる
        path_msg = Path()
        path_msg.header = header
        path_msg.header.frame_id = 'base_link'

        # 30個の滑らかな座標群を、ROSが理解できるPoseStamped型のリストに変換して格納
        path_msg.poses = [PoseStamped(header=path_msg.header, pose=Pose(position=Point(x=float(x_smooth[i]), y=float(y_smooth[i])))) for i in range(len(x_smooth))]

        return path_msg

    def create_path_from_output(self, output: torch.Tensor, header) -> Path:
        """
        補間や平滑化を一切行わない、AIが弾き出した「10個の生データ」だけのPath（経路）を作成します。
        主に開発者がRViz等で想定通りの推論が出ているか確認するためのものです。
        """
        path_msg = Path()
        path_msg.header = header
        path_msg.header.frame_id = 'base_link'
        waypoints = output.cpu().numpy().reshape(-1, 2)

        # 車の「現在地(x=0.0, y=0.0)」を先頭に追加（これから走る軌道の起点を明確にするため）
        path_msg.poses.append(PoseStamped(header=path_msg.header, pose=Pose(position=Point(x=0.0, y=0.0))))
        
        # ⚠️ (注意): 元のコードではここでリストを上書き代入しているため、上の現在地(0,0)が消えてしまっています。
        # AIの出力した10個のウェイポイントを追加します（本来は .extend() などで繋ぐ運用が望ましいです）
        path_msg.poses = [PoseStamped(header=path_msg.header, pose=Pose(position=Point(x=float(x), y=float(y)))) for x, y in waypoints]

        return path_msg


def main(args=None) -> None:
    rclpy.init(args=args)
    node = InferenceNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
