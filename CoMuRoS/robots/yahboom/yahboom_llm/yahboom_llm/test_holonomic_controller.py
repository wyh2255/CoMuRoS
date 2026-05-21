#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全向移动平台位置控制器测试模块。

该模块实现了用于控制全向轮机器人（如 Yahboom 机器人）的位置控制器，包含：
- 基于里程计（Odometry）的机器人位姿反馈
- XY 平面的比例控制
- 偏航角的比例控制
- 发布速度指令到 /r1/cmd_vel 话题
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import math

class HolonomicPositionController(Node):
    """全向移动平台位置控制器节点。

    订阅 /r1/odom 获取机器人当前位姿，通过比例控制器计算控制量，
    并发布速度指令到 /r1/cmd_vel 话题，实现机器人的定点运动控制。
    适用于麦克纳姆轮或其他全向轮底盘。
    """
    def __init__(self):
        """初始化全向移动平台位置控制器节点。

        设置里程计订阅、速度指令发布、控制循环定时器以及比例控制器参数。
        全向轮底盘可独立控制 X、Y 方向的平移和偏航旋转。
        """
        super().__init__('holonomic_position_controller')

        # 订阅里程计话题 /r1/odom，获取机器人当前位姿
        self.create_subscription(Odometry, '/r1/odom', self.odom_callback, 10)

        # 发布速度指令到 /r1/cmd_vel，控制机器人运动
        self.cmd_pub = self.create_publisher(Twist, '/r1/cmd_vel', 10)

        # 控制循环定时器，20 Hz（周期 0.05 秒）
        self.timer = self.create_timer(0.05, self.control_loop)

        # 机器人当前位姿状态（由里程计回调更新）
        # 全向移动平台在平面上运动，仅有 (x, y, yaw)
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0

        # 目标位姿（由 goto() 方法设置）
        self.goal_x = 0.0
        self.goal_y = 0.0
        self.goal_yaw = 0.0

        # 比例增益
        self.kp_xy = 1.0   # XY 平面运动比例增益
        self.kp_yaw = 1.0  # 偏航角比例增益

        self.get_logger().info("全向移动位置控制器启动。")

    def odom_callback(self, msg):
        """里程计回调函数。

        从 /r1/odom 话题中提取机器人的当前位置（x, y）和偏航角（yaw），
        并更新到控制器的内部状态中。偏航角从四元数转换而来。
        全向移动平台在二维平面上运动，仅有 3 个自由度 (x, y, yaw)。
        """
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y

        # 从四元数中提取偏航角（yaw）
        # 转换公式：yaw = atan2(2*(w*z + x*y), 1 - 2*(y^2 + z^2))
        q = msg.pose.pose.orientation
        siny = 2.0 * (q.w*q.z + q.x*q.y)
        cosy = 1.0 - 2.0 * (q.y*q.y + q.z*q.z)
        self.yaw = math.atan2(siny, cosy)

    def control_loop(self):
        """主控制循环，以 20 Hz 频率执行。

        每个周期执行以下步骤：
        1. 计算各通道的位置误差（X、Y、偏航角）
        2. 比例控制律计算各通道速度指令
        3. 对所有输出进行限幅
        4. 发布速度指令到 /r1/cmd_vel

        全向移动平台的优势在于 X 和 Y 方向可以独立控制，互不耦合。
        """
        # 计算位置误差（目标 - 当前位置）
        ex = self.goal_x - self.x       # X 方向误差
        ey = self.goal_y - self.y       # Y 方向误差
        eyaw = self.wrap(self.goal_yaw - self.yaw)  # 偏航角误差（归一化到 [-pi, pi]）

        # 速度指令计算（全向移动比例控制）
        cmd = Twist()
        cmd.linear.x = self.kp_xy * ex   # X 方向线速度
        cmd.linear.y = self.kp_xy * ey   # Y 方向线速度
        cmd.angular.z = self.kp_yaw * eyaw  # 偏航角速度

        # 速度限幅（可选）：将各通道输出限制在 [-1.0, 1.0] 范围
        cmd.linear.x = max(min(cmd.linear.x, 1.0), -1.0)
        cmd.linear.y = max(min(cmd.linear.y, 1.0), -1.0)
        cmd.angular.z = max(min(cmd.angular.z, 1.0), -1.0)

        # 发布速度指令到 /r1/cmd_vel 话题
        self.cmd_pub.publish(cmd)

    def goto(self, x, y, yaw_deg):
        """设置全向移动平台的新目标位姿。

        设置目标位置 (x, y) 和目标偏航角，控制器将驱动机器人移动到该点。
        全向移动平台可以同时独立控制 X 和 Y 方向运动。

        参数:
            x (float): 目标 X 坐标（米）
            y (float): 目标 Y 坐标（米）
            yaw_deg (float): 目标偏航角（度），内部转换为弧度
        """
        self.goal_x = x
        self.goal_y = y
        self.goal_yaw = math.radians(yaw_deg)  # 角度转弧度
        self.get_logger().info(f"新目标设置: ({x}, {y}, {yaw_deg}°)")

    @staticmethod
    def wrap(a):
        """将角度归一化到 [-pi, pi] 范围内。

        使用 atan2(sin(a), cos(a)) 确保角度始终在合法的周期范围内，
        避免角度误差出现跳变（如从 179° 到 -179° 的跃变）。
        """
        return math.atan2(math.sin(a), math.cos(a))


def main(args=None):
    """主函数：启动全向移动平台位置控制器节点。

    示例任务：设置机器人移动到 (-1.0, -1.0) 位置，偏航角 0 度。
    节点启动后会持续运行，直到被外部中断。
    """
    rclpy.init(args=args)
    node = HolonomicPositionController()

    # 示例：发送目标点 (-1.0, -1.0)，偏航角 0 度
    node.goto(-1.0, -1.0, 0.0)

    rclpy.spin(node)    # 保持节点运行，等待回调触发

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
