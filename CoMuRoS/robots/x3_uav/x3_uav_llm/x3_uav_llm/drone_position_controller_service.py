#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
无人机位置控制器服务模块（Drone Position Controller Service）

该模块实现了一个阻塞式的无人机位置控制服务节点。
通过订阅里程计数据（Odometry），采用PID控制算法计算控制指令，
并将速度指令发布到/cmd_vel话题，驱动无人机到达目标位置。

主要功能：
  - 提供GotoPoseDrone服务接口，接收目标位置（x, y, z, yaw）
  - 订阅里程计获取当前位置和姿态
  - 高度通道使用PI控制器（带积分限幅）
  - 水平通道和偏航通道使用P控制器
  - 支持取消指令
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

from robot_interface.srv import GotoPoseDrone

NAMESPACE = 'r3'

class DronePositionController(Node):
    """无人机位置控制器节点"""

    def __init__(self):
        """初始化无人机位置控制器"""
        super().__init__("drone_position_controller")
        self.declare_parameter('namespace', NAMESPACE)
        self.namespace: str = self.get_parameter('namespace').value

        # 回调组：用于支持重入（避免阻塞影响其他回调）
        self.service_cb_group = ReentrantCallbackGroup()
        self.subscription_cb_group = ReentrantCallbackGroup()

        # 订阅里程计数据
        self.create_subscription(
            Odometry, f"/{self.namespace}/odom", self.odom_callback, 10,
            callback_group=self.subscription_cb_group
        )

        # 订阅取消命令话题
        self.create_subscription(
            Bool, f"/{self.namespace}/cancel_goto_pose_goal", self.cancel_callback, 10,
            callback_group=self.subscription_cb_group
        )

        # 发布速度控制指令
        self.cmd_pub = self.create_publisher(Twist, f"/{self.namespace}/cmd_vel", 10)

        # 创建GotoPoseDrone服务（阻塞式回调）
        self.srv = self.create_service(
            GotoPoseDrone,
            f"/{self.namespace}/goto_pose",
            self.goto_service_callback,
            callback_group=self.service_cb_group
        )

        # 机器人状态
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.yaw = 0.0
        self.odom_received = False

        # 取消标志
        self.cancel_requested = False

        # 控制器增益
        self.kp_xy = 1.0       # 水平位置比例增益
        self.kp_yaw = 1.0      # 偏航比例增益
        self.kp_z = 0.35       # 高度比例增益
        self.ki_z = 0.09       # 高度积分增益
        self.integral_z = 0.0  # 高度误差积分累积
        self.integral_limit = 0.5  # 积分限幅

        self.get_logger().info("Blocking Drone Controller Initialized.")

    def odom_callback(self, msg):
        """里程计回调：更新无人机当前位置和偏航角"""
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        self.z = msg.pose.pose.position.z

        # 从四元数计算偏航角
        q = msg.pose.pose.orientation
        siny = 2 * (q.w*q.z + q.x*q.y)
        cosy = 1 - 2 * (q.y*q.y + q.z*q.z)
        self.yaw = math.atan2(siny, cosy)

        self.odom_received = True

    def cancel_callback(self, msg):
        """取消请求回调：收到取消信号时设置取消标志"""
        if msg.data:
            self.cancel_requested = True
            self.get_logger().warn("[Drone] CANCEL request received!")

    # -----------------------------------------------------
    # 阻塞式服务回调 - 核心控制循环
    # -----------------------------------------------------
    def goto_service_callback(self, request, response):
        """
        位置控制服务回调（阻塞式）

        采用PI控制器控制高度，P控制器控制水平和偏航，
        当所有误差低于阈值时返回成功。
        """
        goal_x = request.x
        goal_y = request.y
        goal_z = request.z
        goal_yaw = math.radians(request.yaw_deg)

        self.cancel_requested = False
        self.integral_z = 0.0

        # 等待里程计数据就绪（最多5秒）
        t0 = time.time()
        while not self.odom_received and time.time() - t0 < 5.0:
            time.sleep(0.01)

        if not self.odom_received:
            response.accepted = False
            response.success = False
            response.message = "No odometry data!"
            return response

        # 接受服务请求
        response.accepted = True
        response.success = False
        response.message = "Drone goal execution started."
        self.get_logger().info(response.message)

        timeout = time.time() + 60.0  # 60秒超时
        last_time = time.time()

        while rclpy.ok():
            # 超时检查
            if time.time() > timeout:
                self.stop()
                response.success = False
                response.message = "Drone goal timed out!"
                return response

            # 取消检查
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

            # 计算各通道误差
            ex = goal_x - self.x
            ey = goal_y - self.y
            ez = goal_z - self.z
            eyaw = self.wrap(goal_yaw - self.yaw)

            # ----- 高度PI控制 -----
            self.integral_z += ez * dt
            self.integral_z = max(min(self.integral_z, self.integral_limit), -self.integral_limit)
            vz = self.kp_z * ez + self.ki_z * self.integral_z
            vz = max(min(vz, 1.0), -1.0)

            # XY平面和偏航P控制
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

            # 到达条件：所有误差小于阈值
            if abs(ex) < 0.05 and abs(ey) < 0.05 and abs(ez) < 0.05 and abs(eyaw) < 0.05:
                self.stop()
                response.success = True
                response.message = "Drone reached the goal!"
                self.get_logger().info("[Drone] Goal reached!")
                return response

            time.sleep(0.05)

    def stop(self):
        """安全停止无人机：发布零速度指令"""
        self.cmd_pub.publish(Twist())

    @staticmethod
    def wrap(a):
        """将角度归一化到[-pi, pi]范围"""
        return math.atan2(math.sin(a), math.cos(a))


def main(args=None):
    """节点入口函数：初始化无人机位置控制器"""
    rclpy.init(args=args)
    node = DronePositionController()

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
