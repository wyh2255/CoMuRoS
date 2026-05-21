#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
全向移动机器人位置控制器服务模块

该模块实现了全向驱动（Holonomic Drive）机器人的位置控制服务。
作为ROS2服务端，它订阅里程计话题获取当前位置信息，计算控制指令
并发布到/cmd_vel话题以驱动机器人运动。支持目标取消功能和超时机制。

ROS2接口:
  - 订阅: /{namespace}/odom (Odometry) — 里程计信息
  - 订阅: /{namespace}/cancel_goto_pose_goal (Bool) — 取消请求
  - 发布: /{namespace}/cmd_vel (Twist) — 速度控制指令
  - 服务: /{namespace}/goto_pose (GotoPoseHolonomic) — 位置控制服务
"""

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool

import math
import time

from robot_interface.srv import GotoPoseHolonomic

# 机器人命名空间
NAMESPACE = 'r1'

class HolonomicPositionController(Node):
    """
    全向移动机器人位置控制器

    作为ROS2服务节点，通过比例控制（P控制）实现全向机器人的位置闭环控制。
    接收GotoPoseHolonomic服务请求，读取里程计数据，计算控制误差并输出
    Twist速度指令到/cmd_vel话题。采用阻塞式服务回调模式。

    控制策略:
      - XY方向: 比例控制 (kp_xy = 1.2)
      - 偏航角: 比例控制 (kp_yaw = 2.0)
      - 速度限幅: [-1.0, 1.0] m/s
    """

    def __init__(self):
        """初始化全向位置控制器，设置参数、订阅器、发布器和服务"""
        super().__init__("holonomic_position_controller")
        self.declare_parameter('namespace', NAMESPACE)
        self.namespace: str = self.get_parameter('namespace').value

        # ========== 回调组 ==========
        # 使用可重入回调组，允许服务回调和订阅回调并发执行
        self.service_cb_group = ReentrantCallbackGroup()
        self.subscription_cb_group = ReentrantCallbackGroup()

        # ========== 订阅器 ==========
        # 订阅里程计话题以获取机器人当前位置
        self.create_subscription(
            Odometry,
            f"/{self.namespace}/odom",
            self.odom_callback,
            10,
            callback_group=self.subscription_cb_group
        )
        # 订阅取消话题以支持任务取消
        self.create_subscription(
            Bool,
            f"/{self.namespace}/cancel_goto_pose_goal",
            self.cancel_callback,
            10,
            callback_group=self.subscription_cb_group
        )

        # ========== 发布器 ==========
        # 发布速度控制指令到/cmd_vel话题
        self.cmd_pub = self.create_publisher(Twist, f"/{self.namespace}/cmd_vel", 10)

        # ========== 服务端 ==========
        # 提供位置控制服务，供上层节点（如LLM节点）调用
        self.srv = self.create_service(
            GotoPoseHolonomic,
            f"/{self.namespace}/goto_pose",
            self.goto_service_callback,
            callback_group=self.service_cb_group
        )

        # ========== 机器人状态变量 ==========
        self.x = 0.0              # 当前X位置 (m)
        self.y = 0.0              # 当前Y位置 (m)
        self.yaw = 0.0            # 当前偏航角 (rad)
        self.odom_received = False # 是否已收到里程计数据

        # 取消请求标志
        self.cancel_requested = False

        # ========== 控制增益 ==========
        self.kp_xy = 1.2   # XY方向比例增益
        self.kp_yaw = 2.0  # 偏航角比例增益

        self.get_logger().info("Holonomic Blocking Controller Initialized.")

    # =====================================================
    def odom_callback(self, msg):
        """
        /odom 话题回调函数
        更新机器人当前位置和偏航角（从四元数计算欧拉角）
        """
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y

        # 从四元数计算偏航角
        q = msg.pose.pose.orientation
        siny = 2 * (q.w * q.z + q.x * q.y)
        cosy = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.yaw = math.atan2(siny, cosy)

        self.odom_received = True

    # =====================================================
    def cancel_callback(self, msg):
        """
        /cancel_goto_pose_goal 话题回调函数
        收到取消请求时设置取消标志
        """
        if msg.data:
            self.cancel_requested = True
            self.get_logger().warn("[Holonomic] CANCEL request received!")

    # =====================================================
    # 阻塞式服务回调
    # =====================================================
    def goto_service_callback(self, request, response):
        """
        GoToPoseHolonomic服务回调（阻塞式）

        在服务回调内直接执行控制循环，等待到达目标点或超时/取消才返回。
        使用比例控制计算速度指令并发布到/cmd_vel。

        收敛条件: 位置误差 < 0.05m 且 偏航误差 < 0.05rad
        """
        self.get_logger().info(
            f"[Holonomic] New Goal: ({request.x}, {request.y}, {request.yaw_deg}°)"
        )

        goal_x = request.x
        goal_y = request.y
        goal_yaw = math.radians(request.yaw_deg)

        self.cancel_requested = False

        # 等待里程计数据到达（最多3秒）
        start_wait = time.time()
        while not self.odom_received and time.time() - start_wait < 3.0:
            time.sleep(0.01)

        if not self.odom_received:
            response.accepted = False
            response.success = False
            response.message = "No odometry data received."
            return response

        # 接受服务请求（最终成功状态稍后设置）
        response.accepted = True
        response.success = False
        response.message = "Holonomic goal execution started."
        self.get_logger().info("Starting holonomic movement...")

        # ========== 运动控制循环 ==========
        timeout = time.time() + 100000.0

        while rclpy.ok():

            # --- 超时检测 ---
            if time.time() > timeout:
                self.stop()
                response.success = False
                response.message = "Holonomic goal timed out."
                self.get_logger().warn(response.message)
                return response

            # --- 取消检测 ---
            if self.cancel_requested:
                self.stop()
                response.success = False
                response.message = "Holonomic goal canceled."
                self.get_logger().warn(response.message)
                return response

            # 计算位置和角度误差
            ex = goal_x - self.x          # X方向误差
            ey = goal_y - self.y          # Y方向误差
            eyaw = self.wrap(goal_yaw - self.yaw)  # 偏航角误差（已归一化到[-pi, pi]）

            dist = math.sqrt(ex*ex + ey*ey)

            # 比例控制 + 速度限幅
            cmd = Twist()
            cmd.linear.x = max(min(self.kp_xy * ex, 1.0), -1.0)
            cmd.linear.y = max(min(self.kp_xy * ey, 1.0), -1.0)
            cmd.angular.z = max(min(self.kp_yaw * eyaw, 1.0), -1.0)

            self.cmd_pub.publish(cmd)

            self.get_logger().info(
                f"[Holonomic] ex={ex:.3f}, ey={ey:.3f}, eyaw={eyaw:.3f}, dist={dist:.3f}"
            )

            # --- 到达目标 ---
            if dist < 0.05 and abs(eyaw) < 0.05:
                self.stop()
                response.success = True
                response.message = "Holonomic goal reached!"
                self.get_logger().info("[Holonomic] Goal reached!")
                return response

            time.sleep(0.05)

    # =====================================================
    def stop(self):
        """停止机器人运动（发布零速度指令）"""
        self.cmd_pub.publish(Twist())

    @staticmethod
    def wrap(a):
        """将角度归一化到 [-pi, pi] 范围"""
        return math.atan2(math.sin(a), math.cos(a))


def main(args=None):
    """主入口函数，初始化全向位置控制器节点并开始ROS2事件循环"""
    rclpy.init(args=args)
    node = HolonomicPositionController()

    # 使用多线程执行器允许并发处理服务和订阅回调
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
