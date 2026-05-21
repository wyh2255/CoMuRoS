#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
无人机位置控制器服务模块（简易P控制版本）

该模块是无人机的简化版位置控制器，使用比例控制（P控制）实现无人机的
三维空间位置控制和偏航角控制。相比于三步PID控制版本，此版本采用单循环
同时对位置和偏航进行控制，Z轴使用PI控制以消除高度静差。

ROS2接口:
  - 订阅: /{namespace}/odom (Odometry) — 里程计信息
  - 订阅: /{namespace}/cancel_goto_pose_goal (Bool) — 取消请求
  - 发布: /{namespace}/cmd_vel (Twist) — 速度控制指令
  - 服务: /{namespace}/goto_pose (GotoPoseDrone) — 位置控制服务
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

from robot_interface.srv import GotoPoseDrone

# 无人机机器人命名空间
NAMESPACE = 'r3'

class DronePositionController(Node):
    """
    无人机位置控制器（简易P控制版本，带Z轴PI控制）

    使用比例控制器（P控制）对X/Y/偏航进行控制，对Z轴使用PI控制
    （比例+积分）以消除高度静差。采用单循环同时控制所有轴。

    控制策略:
      - X/Y: 比例控制 (kp_xy = 1.0)
      - 偏航: 比例控制 (kp_yaw = 1.0)
      - Z轴: PI控制 (kp_z = 0.35, ki_z = 0.09)，带抗积分饱和
      - 收敛条件: 所有轴误差 < 0.05 (位置m/偏航rad)
    """

    def __init__(self):
        """
        初始化无人机位置控制器

        配置ROS2接口（订阅器、发布器、服务端）和控制参数。
        使用可重入回调组以支持并发回调处理。
        """
        super().__init__("drone_position_controller")
        self.declare_parameter('namespace', NAMESPACE)
        self.namespace: str = self.get_parameter('namespace').value

        # ========== 回调组 ==========
        self.service_cb_group = ReentrantCallbackGroup()
        self.subscription_cb_group = ReentrantCallbackGroup()

        # ========== 订阅器 ==========
        # 订阅里程计话题以获取无人机当前位置
        self.create_subscription(
            Odometry, f"/{self.namespace}/odom", self.odom_callback, 10,
            callback_group=self.subscription_cb_group
        )

        # 订阅取消话题以支持任务取消
        self.create_subscription(
            Bool, f"/{self.namespace}/cancel_goto_pose_goal", self.cancel_callback, 10,
            callback_group=self.subscription_cb_group
        )

        # ========== 发布器 ==========
        # 发布速度控制指令到/cmd_vel话题
        self.cmd_pub = self.create_publisher(Twist, f"/{self.namespace}/cmd_vel", 10)

        # ========== 服务端 ==========
        # 提供位置控制服务，供上层节点（如LLM节点）调用
        self.srv = self.create_service(
            GotoPoseDrone,
            f"/{self.namespace}/goto_pose",
            self.goto_service_callback,
            callback_group=self.service_cb_group
        )

        # ========== 无人机状态变量 ==========
        self.x = 0.0              # 当前X位置 (m)
        self.y = 0.0              # 当前Y位置 (m)
        self.z = 0.0              # 当前Z位置/高度 (m)
        self.yaw = 0.0            # 当前偏航角 (rad)
        self.odom_received = False # 是否已收到里程计数据

        # 取消请求标志
        self.cancel_requested = False

        # ========== 控制增益参数 ==========
        self.kp_xy = 1.0   # XY方向比例增益
        self.kp_yaw = 1.0  # 偏航角比例增益
        self.kp_z = 0.35   # Z轴比例增益（较小防止高度震荡）
        self.ki_z = 0.09   # Z轴积分增益（消除高度静差）
        self.integral_z = 0.0      # Z轴积分项累积值
        self.integral_limit = 0.5  # 积分限幅（抗积分饱和）

        self.get_logger().info("Blocking Drone Controller Initialized.")

    # -----------------------------------------------------
    def odom_callback(self, msg):
        """
        /odom 话题回调函数

        更新无人机当前位置（x, y, z）和偏航角（从四元数计算欧拉角）。
        """
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        self.z = msg.pose.pose.position.z

        # 从四元数计算偏航角
        q = msg.pose.pose.orientation
        siny = 2 * (q.w*q.z + q.x*q.y)
        cosy = 1 - 2 * (q.y*q.y + q.z*q.z)
        self.yaw = math.atan2(siny, cosy)

        self.odom_received = True

    # -----------------------------------------------------
    def cancel_callback(self, msg):
        """
        /cancel_goto_pose_goal 话题回调函数

        收到取消请求时设置取消标志，用于中断正在执行的运动控制循环。
        """
        if msg.data:
            self.cancel_requested = True
            self.get_logger().warn("[Drone] CANCEL request received!")

    # -----------------------------------------------------
    # 阻塞式服务回调
    # -----------------------------------------------------
    def goto_service_callback(self, request, response):
        """
        GotoPoseDrone服务回调（阻塞式单循环控制）

        在一个控制循环中同时对所有轴（X, Y, Z, 偏航）进行控制。
        X/Y使用比例控制，Z轴使用PI控制（带抗积分饱和），偏航使用比例控制。

        收敛条件: 位置误差 < 0.05m 且 偏航误差 < 0.05rad
        超时时间: 60秒
        """
        # 解析服务请求中的目标值
        goal_x = request.x
        goal_y = request.y
        goal_z = request.z
        goal_yaw = math.radians(request.yaw_deg)

        self.cancel_requested = False
        self.integral_z = 0.0  # 重置Z轴积分项

        # 等待里程计数据到达（最多5秒）
        t0 = time.time()
        while not self.odom_received and time.time() - t0 < 5.0:
            time.sleep(0.01)

        if not self.odom_received:
            response.accepted = False
            response.success = False
            response.message = "No odometry data!"
            return response

        # 接受服务请求（最终成功状态稍后设置）
        response.accepted = True
        response.success = False
        response.message = "Drone goal execution started."
        self.get_logger().info(response.message)

        # 设置超时时间（60秒）
        timeout = time.time() + 60.0
        last_time = time.time()

        while rclpy.ok():

            # 超时检测
            if time.time() > timeout:
                self.stop()
                response.success = False
                response.message = "Drone goal timed out!"
                return response

            # 取消检测
            if self.cancel_requested:
                self.stop()
                response.success = False
                response.message = "Drone goal canceled!"
                return response

            # 计算时间步长
            now = time.time()
            dt = now - last_time
            last_time = now
            if dt <= 0:
                dt = 0.01

            # 计算各轴误差
            ex = goal_x - self.x
            ey = goal_y - self.y
            ez = goal_z - self.z
            eyaw = self.wrap(goal_yaw - self.yaw)

            # ----------------------
            # Z轴PI控制（带抗积分饱和）
            # ----------------------
            self.integral_z += ez * dt
            self.integral_z = max(min(self.integral_z, self.integral_limit), -self.integral_limit)

            vz = self.kp_z * ez + self.ki_z * self.integral_z
            vz = max(min(vz, 1.0), -1.0)

            # XY轴比例控制 + 偏航比例控制（带输出限幅）
            vx = max(min(self.kp_xy * ex, 1.0), -1.0)
            vy = max(min(self.kp_xy * ey, 1.0), -1.0)
            wz = max(min(self.kp_yaw * eyaw, 1.0), -1.0)

            # 发布速度指令
            cmd = Twist()
            cmd.linear.x = vx
            cmd.linear.y = vy
            cmd.linear.z = vz
            cmd.angular.z = wz
            self.cmd_pub.publish(cmd)

            # 检查是否到达目标（所有轴误差低于阈值）
            if abs(ex) < 0.05 and abs(ey) < 0.05 and abs(ez) < 0.05 and abs(eyaw) < 0.05:
                self.stop()
                response.success = True
                response.message = "Drone reached the goal!"
                self.get_logger().info("[Drone] Goal reached!")
                return response

            time.sleep(0.05)

    # -----------------------------------------------------
    def stop(self):
        """停止无人机运动（发布零速度指令）"""
        self.cmd_pub.publish(Twist())

    @staticmethod
    def wrap(a):
        """将角度归一化到 [-pi, pi] 范围"""
        return math.atan2(math.sin(a), math.cos(a))


def main(args=None):
    """
    主入口函数

    初始化无人机P控制位置控制器节点并开始ROS2事件循环。
    使用多线程执行器以允许并发处理服务和订阅回调。
    """
    rclpy.init(args=args)
    node = DronePositionController()

    # 使用多线程执行器处理并发回调
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
