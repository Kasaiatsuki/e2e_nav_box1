# e2e_nav_box1

ZEDカメラ（RGB画像）を入力とし、End-to-End学習（AI）によって生成された予測軌道をPure Pursuit制御で追従走行する自律移動ロボット向けROS 2パッケージです。
キャンパス環境などの白線がない道での走行を想定し、画像から直接走行経路（Waypoint）を推論して自律走行（icart等）を実現します。

---

## 💻 動作環境 (Environment)
- **OS:** Ubuntu 22.04 LTS
- **ROS 2:** Humble Hawksbill
- **カメラ:** ZED Camera (ZED SDK 対応)
- **言語:** Python 3.10 / C++17
- **GPU:** NVIDIA GPU (CUDA対応、PyTorch実行用)

### 主要な依存ライブラリ
- `torch`, `torchvision` (PyTorch)
- `opencv-python` (cv2)
- `scipy` (B-Spline補間用)
- `pyzed` (ZED SDK Python API - *直通モード使用時のみ*)
- その他のROS 2標準パッケージ (`rclcpp`, `rclpy`, `sensor_msgs`, `geometry_msgs`, `nav_msgs`, `cv_bridge`)

---

## 🏗️ システム構成 (System Architecture)

本パッケージは大きく分けて4つのシステム（ノード/スクリプト）で構成されています。

1. **データ収集 (`create_data.py`)**
   - 人間の操作（ジョイスティック）による走行を記録するノードです。
   - ZEDカメラのRGB画像と、オドメトリ（Pose）を同期させ、一定間隔の未来のWaypoint（軌道）を計算して `data/` フォルダへ保存・蓄積します。
   - ジョイコントローラの特定ボタン（デフォルトは `[1]`）で収集の開始/一時停止をトグルできます。

2. **AIモデル (`network.py`)**
   - RGB画像（3チャンネル, 64x48）を受け取り、10個のWaypoint（x, y座標）を出力する3層CNN＋2層全結合層の推論用ネットワークです。

3. **モデル学習 (`train.py`)**
   - `create_data.py` で集めた走行データを使ってAIを訓練するスクリプトです。
   - 誤差を計算し、推論モデルの重みファイル (`e2e_model.pt`) を生成して `weights/` フォルダに保存します。

4. **推論＆経路追従 (`inference_node.py` + `pure_pursuit_node.cpp`)**
   - **推論ノード**: リアルタイムでカメラ画像を受け取り、学習済みAIで未来のWaypoint群を推論します。出力されたカクカクの点群をSciPyを用いてB-スプライン曲線で自然に補間し、`/e2e_planner/path` として配信します。
   - **追従ノード**: `/e2e_planner/path` を受け取り、前方注視モデル（Pure Pursuitアルゴリズム）に基づいて適切なアクセル（リニア）とハンドル角（アンギュラ）を計算し、ロボットのドライバ（`ypspur_ros`等）へ向けて `/cmd_vel` を送信します。

---

## 🚀 インストール手順 (Installation)

### 1. ワークスペースへのクローン
```bash
cd ~/ros2_ws/src
git clone git@github.com:Kasaiatsuki/e2e_nav_box1.git
```

### 2. 依存関係のインストール
ZED SDK や PyTorch などの必要なPythonライブラリをインストールしてください。
```bash
pip3 install torch torchvision opencv-python scipy pymap3d
# ZED SDKを利用する場合は、別途公式からSDKとPyZEDをインストールしてください。
```

### 3. パッケージのビルド
```bash
cd ~/ros2_ws
colcon build --packages-select e2e_nav_box1
source install/setup.bash
```

---

## 🏃 実行手順 (Usage)

### Step 1: 走行データの収集
実際の環境でジョイスティックを使ってロボットを走らせ、学習データを集めます。
```bash
ros2 run e2e_nav_box1 create_data.py
```
> ※ ジョイコントローラのトグルボタン（デフォルト `1`）を押すと収集の開始/停止が切り替わります。

### Step 2: モデルの学習
収集したデータセットが保存されたディレクトリを指定し、AIモデルを訓練します。
```bash
# 例: dataフォルダ内のデータセットを指定
ros2 run e2e_nav_box1 train.py /path/to/your/dataset/directory
```
> ※ 学習が完了すると `weights/e2e_model.pt` としてモデルが保存されます。

### Step 3: 自律走行の実行（推論 ＋ 制御）
学習が終わったら、推論ノードとモータ制御ノードを立ち上げてAIに運転を任せます。
```bash
# 1. AIの推論（脳）を起動
ros2 run e2e_nav_box1 inference_node.py

# 2. 自動運転モードのフラグをONにする
ros2 topic pub /autonomous std_msgs/msg/Bool "{data: true}" -1

# 3. モーターの制御（手足）を起動
ros2 run e2e_nav_box1 pure_pursuit_node
```

> ⚠️ 注意: 走行中は必ずロボットの緊急停止スイッチを持った状態でテストを行ってください。

---

## 📜 ライセンス (License)

このソフトウェアパッケージは、3条項BSDライセンスの下、再頒布および使用が許可されます。
© 2026 Atsuki Kasai
