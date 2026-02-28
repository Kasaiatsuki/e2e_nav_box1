#include "e2e_nav_box1/pure_pursuit_node.hpp"

namespace e2e_nav_box1 {

PurePursuit::PurePursuit(const rclcpp::NodeOptions & options) : PurePursuit("", options) {}

PurePursuit::PurePursuit(const std::string & name_space, const rclcpp::NodeOptions & options)
: rclcpp::Node("pure_pursuit_node", name_space, options), autonomous_flag_(false)
{
    // パラメータの宣言と取得
    this->declare_parameter("linear_max_vel", 0.5);
    this->declare_parameter("angular_max_vel", 1.0);
    this->declare_parameter("lookahead_distance", 1.0);
    this->declare_parameter("curvature_gain", 1.0);

    linear_max_vel_ = this->get_parameter("linear_max_vel").as_double();
    angular_max_vel_ = this->get_parameter("angular_max_vel").as_double();
    lookahead_distance_ = this->get_parameter("lookahead_distance").as_double();
    curvature_gain_ = this->get_parameter("curvature_gain").as_double();

    // QoSの設定。センサーデータや重要な制御コマンドに適した設定
    auto qos = rclcpp::QoS(rclcpp::KeepLast(10));

    // 自動運転モードのON/OFFフラグを受信するための設定
    subscription_autonomous_ = this->create_subscription<std_msgs::msg::Bool>(
        "/autonomous", qos, std::bind(&PurePursuit::autonomous_callback, this, std::placeholders::_1)
    );

    // E2E Planner（推論ノード）から配信される生成パス（Waypoint群）を購読する設定
    subscription_path_ = this->create_subscription<nav_msgs::msg::Path>(
        "e2e_planner/path", qos, std::bind(&PurePursuit::subscriber_callback_path, this, std::placeholders::_1)
    );

    // 計算したアクセルとハンドルの命令値（cmd_vel）を車両のモーター制御コントローラーへ送信する設定
    // icart (ypspur_ros等)が受け取るトピック名に合わせて "cmd_vel" にする
    publisher_vel_ = this->create_publisher<geometry_msgs::msg::Twist>("cmd_vel", qos);

    RCLCPP_INFO(this->get_logger(), "PurePursuit node initialized. lookahead=%.2f, v_max=%.2f, w_max=%.2f", 
                lookahead_distance_, linear_max_vel_, angular_max_vel_);
}

void PurePursuit::autonomous_callback(const std_msgs::msg::Bool::SharedPtr msg) {
    autonomous_flag_ = msg->data;
    if (!autonomous_flag_) {
        // 自動運転モードがOFFにされた瞬間、確実に停止命令を送る
        geometry_msgs::msg::Twist stop_cmd;
        publisher_vel_->publish(stop_cmd);
        RCLCPP_INFO(this->get_logger(), "Autonomous mode OFF - Stopping robot.");
    } else {
        RCLCPP_INFO(this->get_logger(), "Autonomous mode ON - Resuming control.");
    }
}

void PurePursuit::subscriber_callback_path(const nav_msgs::msg::Path::SharedPtr msg) {
    // 自動運転モードがOFFの時は何もせず終了
    if (!autonomous_flag_) {
        return;
    }

    geometry_msgs::msg::Twist command; // 送信する速度・角速度コマンド用の変数

    // 受信したパスが空っぽ、あるいは異常なデータの場合は安全のために停止命令（ゼロ速度）を送信
    if (!msg || msg->poses.empty()) {
        RCLCPP_WARN_THROTTLE(
            this->get_logger(),
            *this->get_clock(),
            2000,
            "Received empty path. Publishing zero velocity."
        );
        publisher_vel_->publish(command);
        return;
    }

    // --- 1. 目標点（ルックアヘッド・ポイント）の探索 ---
    // 車両の中心座標系（base_link基準）で表現されたパスの中から、
    // 前方注視距離（lookahead_distance）以上離れている最初の点を探し出す
    auto target_it = std::find_if(
        msg->poses.begin(),
        msg->poses.end(),
        [this](const geometry_msgs::msg::PoseStamped& pose) {
            const double dx = pose.pose.position.x;
            const double dy = pose.pose.position.y;
            // 車からその点までの直線距離がルックアヘッド距離以上か判定
            return std::hypot(dx, dy) >= lookahead_distance_;
        }
    );

    // もし全ての点がルックアヘッド距離より手前だった場合（急カーブ後や終点付近など）は、一番最後の点を目標にする
    if (target_it == msg->poses.end()) {
        target_it = std::prev(msg->poses.end());
    }

    // 決定した目標座標 (target_x, target_y) を取り出し、現在地からの直線距離を計算
    const double target_x = target_it->pose.position.x;
    const double target_y = target_it->pose.position.y;
    const double distance = std::hypot(target_x, target_y);

    // ピタリと目標に到着していたり、計算が破綻するほど近い場合は停止
    if (distance < 1e-6) {
        publisher_vel_->publish(command);
        return;
    }

    // --- 2. 速度制限制御 ---
    // 距離が近すぎる時のゼロ割り防止用
    const double safe_lookahead = std::max(lookahead_distance_, 1e-3);
    // 目標が遠ければ 1.0(フルスロットル)、近ければ 0.0 に近づくスケール係数（ブレーキ代わり）
    const double linear_scale = std::clamp(distance / safe_lookahead, 0.0, 1.0);
    // 最大速度にスケール係数を掛け、最終的な前進速度（アクセル）を決定
    const double linear_velocity = std::clamp(linear_max_vel_ * linear_scale, 0.0, linear_max_vel_);
    
    // --- 3. 旋回（ステアリング）量計算（ピュア・パースートの核心部） ---
    // 目標の横ずれ(target_y)をもとに、そこへ到達するための滑らかな円弧の曲がり具合（曲率）を算出
    const double curvature = (target_y * curvature_gain_) / (distance * distance);
    // 算出した曲率と前進速度を掛け合わせて、必要なハンドルの角速度（angular_velocity）に変換
    double angular_velocity = linear_velocity * curvature;

    // 安全のため、算出した角速度が制限値を超えないように切り詰める
    angular_velocity = std::clamp(angular_velocity, -angular_max_vel_, angular_max_vel_);

    // --- 4. 制御値の送信 ---
    command.linear.x = linear_velocity;  // アクセルの踏み込み量（前進速度）
    command.angular.z = angular_velocity; // ハンドルの切れ角（旋回速度）
    publisher_vel_->publish(command);     // 実際の車両へモーター指令を送信
}

}  // namespace e2e_nav_box1

// ==========================================
// ROS 2 エントリポイント
// ==========================================
int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<e2e_nav_box1::PurePursuit>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
