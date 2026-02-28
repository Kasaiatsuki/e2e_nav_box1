#ifndef E2E_NAV_BOX1__PURE_PURSUIT_NODE_HPP_
#define E2E_NAV_BOX1__PURE_PURSUIT_NODE_HPP_

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <nav_msgs/msg/path.hpp>
#include <std_msgs/msg/bool.hpp>
#include <cmath>
#include <algorithm>

namespace e2e_nav_box1 {

/**
 * @brief パスを追従するPure PursuitのROS 2ノード
 * 
 * E2Eプランナーから受信した経路（Path）をもとに、ロボット（icart）を制御する
 * 速度コマンド（Twist）を計算して送信します。
 */
class PurePursuit : public rclcpp::Node {
public:
    /**
     * @brief Nodeコンストラクタ。各種パラメータやPub/Subを初期化します。
     * @param options ROS 2 ノード初期化のオプション
     */
    explicit PurePursuit(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
    
    /**
     * @brief 名前空間指定用のNodeコンストラクタ。
     * @param name_space ノードの名前空間
     * @param options ROS 2 ノード初期化のオプション
     */
    PurePursuit(const std::string & name_space, const rclcpp::NodeOptions & options = rclcpp::NodeOptions());

private:
    // --- パラメータ ---
    double linear_max_vel_;      ///< [m/s] 直進の最高速度制限
    double angular_max_vel_;     ///< [rad/s] 旋回（ハンドル）の最大角速度制限
    double lookahead_distance_;  ///< [m] 前方注視距離（追跡する目標点までの最小距離）
    double curvature_gain_;      ///< [-] 曲率ゲイン（曲がり具合を計算する係数）
    bool autonomous_flag_;       ///< [-] 自動運転モードのON/OFF状態

    // --- Publisher / Subscriber ---
    rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr subscription_path_;       ///< 推論ノードからの生成パスを受信
    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr subscription_autonomous_; ///< 自動運転の切り替え信号を受信
    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr publisher_vel_;        ///< モーター制御用(cmd_vel)の送信

    /**
     * @brief 自動運転モードを切り替えるコールバック関数
     * @param msg true: 自動運転ON, false: 手動運転(OFF)
     */
    void autonomous_callback(const std_msgs::msg::Bool::SharedPtr msg);

    /**
     * @brief E2Eプランナーから送られてきた経路を受信し、追従制御計算を行うコールバック関数
     * @param msg 受信したPathメッセージ(PoseStampedの配列)
     */
    void subscriber_callback_path(const nav_msgs::msg::Path::SharedPtr msg);
};

}  // namespace e2e_nav_box1

#endif  // E2E_NAV_BOX1__PURE_PURSUIT_NODE_HPP_
