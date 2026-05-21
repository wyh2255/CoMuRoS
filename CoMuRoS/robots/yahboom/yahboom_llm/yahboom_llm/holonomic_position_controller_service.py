#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全向机器人位置控制器服务模块（Holonomic Position Controller Service）

该模块实现了一个阻塞式的全向移动机器人位置控制服务节点。
通过订阅里程计数据（Odometry），采用P控制算法计算控制指令，
并将速度指令发布到/cmd_vel话题，驱动机器人到达目标位置。

主要功能：
  - 提供GotoPoseHolonomic服务接口，接收目标位置（x, y, yaw）
  - 订阅里程计获取当前位置和姿态
  - 水平和偏航通道使用P控制器
  - 支持取消指令（通过cancel话题）
  - 到达目标精度后自动停止
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

NAMESPACE = 'r2'

class HolonomicPositionController(Node):
    """全向移动机器人位置控制器节点"""

    def __init__(self):
        """初始化全向机器人位置控制器"""
        super().__init__("holonomic_position_controller")
        self.declare_parameter('namespace', NAMESPACE)
        self.namespace: str = self.get_parameter('namespace').value

        # 回调组
        self.service_cb_group = ReentrantCallbackGroup()
        self.subscription_cb_group = ReentrantCallbackGroup()

        # 订阅里程计数据
        self.create_subscription(
            Odometry,
            f"/{self.namespace}/odom",
            self.odom_callback,
            10,
            callback_group=self.subscription_cb_group
        )

        # 订阅取消命令话题
        self.create_subscription(
            Bool,
            f"/{self.namespace}/cancel_goto_pose_goal",
            self.cancel_callback,
            10,
            callback_group=self.subscription_cb_group
        )

        # 发布速度控制指令
        self.cmd_pub = self.create_publisher(Twist, f"/{self.namespace}/cmd_vel", 10)

        # 创建GotoPoseHolonomic服务（阻塞式回调）
        self.srv = self.create_service(
            GotoPoseHolonomic,
            f"/{self.namespace}/goto_pose",
            self.goto_service_callback,
            callback_group=self.service_cb_group
        )

        # 机器人状态
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.odom_received = False

        # 取消标志
        self.cancel_requested = False

        # 控制器增益
        self.kp_xy = 1.2    # 水平位置比例增益
        self.kp_yaw = 2.0   # 偏航比例增益

        self.get_logger().info("Holonomic Blocking Controller Initialized.")

    def odom_callback(self, msg):
        """里程计回调：更新机器人当前位置和偏航角"""
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y

        # 从四元数计算偏航角
        q = msg.pose.pose.orientation
        siny = 2 * (q.w * q.z + q.x * q.y)
        cosy = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.yaw = math.atan2(siny, cosy)

        self.odom_received = True

    def cancel_callback(self, msg):
        """取消请求回调：收到取消信号时设置取消标志"""
        if msg.data:
            self.cancel_requested = True
            self.get_logger().warn("[Holonomic] CANCEL request received!")

    # -----------------------------------------------------
    # 阻塞式服务回调 - 核心控制循环
    # -----------------------------------------------------
    def goto_service_callback(self, request, response):
        """
        位置控制服务回调（阻塞式）

        采用P控制器控制水平和偏航，
        当距离和角度误差均低于阈值时返回成功。
        """
        self.get_logger().info(
            f"[Holonomic] New Goal: ({request.x}, {request.y}, {request.yaw_deg}°"
        )

        goal_x = request.x
        goal_y = request.y
        goal_yaw = math.radians(request.yaw_deg)

        self.cancel_requested = False

        # 等待里程计数据就绪（最多3秒）
        start_wait = time.time()
        while not self.odom_received and time.time() - start_wait < 3.0:
            time.sleep(0.01)

        if not self.odom_received:
            response.accepted = False
            response.success = False
            response.message = "No odometry data received."
            return response

        # 接受服务请求（最终结果稍后返回）
        response.accepted = True
        response.success = False
        response.message = "Holonomic goal execution started."
        self.get_logger().info("Starting holonomic movement...")

        # 运动控制循环
        timeout = time.time() + 100000.0

        while rclpy.ok():
            # --- 超时检查 ---
            if time.time() > timeout:
                self.stop()
                response.success = False
                response.message = "Holonomic goal timed out."
                self.get_logger().warn(response.message)
                return response

            # --- 取消检查 ---
            if self.cancel_requested:
                self.stop()
                response.success = False
                response.message = "Holonomic goal canceled."
                self.get_logger().warn(response.message)
                return response

            # 计算各通道误差
            ex = goal_x - self.x
            ey = goal_y - self.y
            eyaw = self.wrap(goal_yaw - self.yaw)
            dist = math.sqrt(ex*ex + ey*ey)

            # P控制器：速度与误差成比例
            cmd = Twist()
            cmd.linear.x = max(min(self.kp_xy * ex, 1.0), -1.0)
            cmd.linear.y = max(min(self.kp_xy * ey, 1.0), -1.0)
            cmd.angular.z = max(min(self.kp_yaw * eyaw, 1.0), -1.0)

            self.cmd_pub.publish(cmd)

            self.get_logger().info(
                f"[Holonomic] ex={ex:.3f}, ey={ey:.3f}, eyaw={eyaw:.3f}, dist={dist:.3f}"
            )

            # --- 到达目标条件 ---
            if dist < 0.05 and abs(eyaw) < 0.05:
                self.stop()
                response.success = True
                response.message = "Holonomic goal reached!"
                self.get_logger().info("[Holonomic] Goal reached!")
                return response

            time.sleep(0.05)

    def stop(self):
        """安全停止机器人：发布零速度指令"""
        self.cmd_pub.publish(Twist())

    @staticmethod
    def wrap(a):
        """将角度归一化到[-pi, pi]范围"""
        return math.atan2(math.sin(a), math.cos(a))


def main(args=None):
    """节点入口函数：初始化全向机器人位置控制器"""
    rclpy.init(args=args)
    node = HolonomicPositionController()

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
