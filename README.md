# e2e_nav_box1

ZEDカメラのRGB画像を入力とし、End-to-End学習（模倣学習）によって自律走行を行うROS 2パッケージです。
人間の手動操作を模倣するBehavioral Cloningアプローチを採用しており、画像から直接角速度（ステアリング）を推論します。

---

## 💻 動作環境 (Environment)
- **OS:** Ubuntu 20.04 LTS
- **ROS 2:** Foxy Fitzroy
- **カメラ:** ZED Camera (ZED SDK 対応) ※ HD720 (1280x720) / 640x360 にリサイズして使用
- **言語:** Python 3.8
- **GPU:** NVIDIA GPU (CUDA対応、PyTorch実行用)

### 主要な依存ライブラリ
```bash
pip3 install torch torchvision opencv-python schedulefree
# ZED SDKを別途公式からインストールしてください
```

---

## 🏗️ システム構成 (System Architecture)

```
[ZED Camera] → [create_data.py] → [data/] → [train.py] → [weights/e2e_model.pt]
                                                                    ↓
[ZED Camera] ─────────────────────────────────────── [inference_node.py] → /cmd_vel → [icart_driver]
```

| スクリプト | 役割 |
|---|---|
| `create_data.py` | ジョイスティック手動走行中の画像と角速度を記録 |
| `network.py` | CNN + 全結合層のニューラルネットワーク定義 |
| `train.py` | 収集データを使ってモデルを学習 |
| `inference_node.py` | 学習済みモデルで画像から角速度をリアルタイム推論し `/cmd_vel` を配信 |

---

## 🚀 インストール手順 (Installation)

### 1. ワークスペースへのクローン
```bash
cd ~/my_ws/src
git clone <repository_url> e2e_nav_box1
```

### 2. 依存関係のインストール
```bash
pip3 install torch torchvision opencv-python schedulefree
```

### 3. パッケージのビルド
```bash
cd ~/my_ws
colcon build --packages-select e2e_nav_box1
source install/setup.bash
```

---

## 🏃 実行手順 (Usage)

### Step 1: ロボットドライバの起動

**手動操作（データ収集）時：**
```bash
ros2 launch icart_driver icart_drive.launch.py
```

**自律走行（推論）時：**
```bash
ros2 launch icart_driver icart_inference.launch.py
```

### Step 2: 走行データの収集

別ターミナルで実行：
```bash
ros2 run e2e_nav_box1 create_data
```

- ジョイスティックでロボットを走らせながらデータを収集します
- `RBボタン`（`axes[2]`）でデータ収集の開始/一時停止をトグルできます
- **`Ctrl+C` は1回だけ押し**、「🔵Saved」ログが出るまで待ってください（途中で再度 `Ctrl+C` すると保存が中断されデータが欠損します）

収集されたデータは以下に保存されます：
```
~/my_ws/install/e2e_nav_box1/lib/python3.8/site-packages/data/<タイムスタンプ>_dataset/
├── images/     # 640x360 にリサイズ済み PNG 画像
└── angular_vel/ # 対応する角速度 CSV
```

### Step 3: モデルの学習

```bash
cd ~/my_ws/src/e2e_nav_box1/e2e_nav_box1
python3 train.py ~/kasai_ws/install/e2e_nav_box1/lib/python3.8/site-packages/data/<タイムスタンプ>_dataset
```

- 学習完了後、モデルは `~/my_ws/src/e2e_nav_box1/weights/e2e_model.pt` に保存されます
- 学習後は必ずビルドして `install` に反映させてください：
  ```bash
  cd ~/my_ws && colcon build --packages-select e2e_nav_box1
  ```

### Step 4: 自律走行（推論）

```bash
# ターミナル1: ドライバのみ起動（コントローラーノードなし）
ros2 launch icart_driver icart_inference.launch.py

# ターミナル2: 推論ノード起動
ros2 run e2e_nav_box1 inference_node
```

> ⚠️ 走行中は必ずロボットの緊急停止スイッチを持った状態でテストを行ってください。

---

## ⚙️ 設定ファイル (Configuration)

### `config/train.yaml`
| パラメータ | 説明 | デフォルト値 |
|---|---|---|
| `epochs` | 学習エポック数 | 200 |
| `batch_size` | バッチサイズ | 8 |
| `learning_rate` | 学習率 | 0.0002 |
| `num_workers` | DataLoaderのプロセス数 | 2 |
| `weight_file` | 出力モデルファイル名 | `e2e_model.pt` |

### `icart_driver/config/main_param.yaml`
| パラメータ | 説明 | デフォルト値 |
|---|---|---|
| `linear_max.vel` | 最大並進速度 (m/s) | 0.5 |
| `angular_max.vel` | 最大角速度 (rad/s) | 1.0 |

---

## 🙏 謝辞 (Acknowledgements)

本プロジェクトを開発するにあたり、以下のオープンソースリポジトリ・プロジェクトを参考にさせていただきました。
- [aiformula](https://github.com/open-rdc/aiformula/tree/feat/e2e)
- [e2enav](https://github.com/kyo0221/e2enav)

---

## 📜 ライセンス (License)

- このソフトウェアパッケージは、3条項BSDライセンスの下、再頒布および使用が許可されます。
- © 2026 Atsuki Kasai
