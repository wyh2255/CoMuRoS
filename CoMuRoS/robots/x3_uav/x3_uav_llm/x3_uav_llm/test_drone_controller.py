#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
无人机位置控制器测试模块。

该模块实现了用于控制四旋翼无人机的位置控制器，包含：
- 基于里程计（Odometry）的无人机位姿反馈
- XY平面和偏航角的比例控制
- 高度方向的PI控制（带抗积分饱和）
- 发布速度指令到 /r3/cmd_vel 话题
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import math
import time

class DronePositionController(Node):
    """无人机位置控制器节点。

    订阅 /r3/odom 获取无人机当前位姿，通过 PI 控制器计算控制量，
    并发布速度指令到 /r3/cmd_vel 话题，实现对无人机的定点悬停和位置控制。
    """
    def __init__(self):
        """初始化无人机位置控制器节点。

        设置里程计订阅、速度指令发布、控制循环定时器以及控制器参数。
        高度通道采用 PI 控制器，XY 平面和偏航采用比例控制器。
        """
        super().__init__('drone_position_controller')

        # 订阅里程计话题 /r3/odom，获取无人机当前位姿
        self.create_subscription(Odometry, '/r3/odom', self.odom_callback, 10)

        # 发布速度指令到 /r3/cmd_vel，控制无人机运动
        self.cmd_pub = self.create_publisher(Twist, '/r3/cmd_vel', 10)

        # 控制循环定时器，20 Hz（周期 0.05 秒）
        self.timer = self.create_timer(0.05, self.control_loop)

        # 无人机当前位姿状态（由里程计回调更新）
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.yaw = 0.0

        # 目标位姿（由 goto() 方法设置）
        self.goal_x = 0.0
        self.goal_y = 0.0
        self.goal_z = 0.0
        self.goal_yaw = 0.0

        # 比例增益：XY 平面和偏航角
        self.kp_xy = 1.0
        self.kp_yaw = 1.0

        # 高度通道 PI 控制器参数
        self.kp_z = 0.35         # 比例增益
        self.ki_z = 0.05         # 积分增益
        self.integral_z = 0.0    # 积分项累积值
        self.integral_limit = 0.5  # 抗积分饱和上限

        # 时间记录（用于计算控制周期 dt）
        self.last_time = time.time()

        self.get_logger().info("无人机位置控制器启动（含高度 PI 控制）。")

    def odom_callback(self, msg):
        """里程计回调函数。

        从 /r3/odom 话题中提取无人机的当前位置（x, y, z）和偏航角（yaw），
        并更新到控制器的内部状态中。偏航角从四元数转换而来。
        """
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        self.z = msg.pose.pose.position.z

        # 从四元数中提取偏航角（yaw）
        # 转换公式：yaw = atan2(2*(w*z + x*y), 1 - 2*(y^2 + z^2))
        q = msg.pose.pose.orientation
        siny = 2.0 * (q.w*q.z + q.x*q.y)
        cosy = 1.0 - 2.0 * (q.y*q.y + q.z*q.z)
        self.yaw = math.atan2(siny, cosy)

    def control_loop(self):
        """主控制循环，以 20 Hz 频率执行。

        每个周期执行以下步骤：
        1. 计算时间步长 dt
        2. 计算各通道的位置误差
        3. 高度通道：PI 控制（带抗积分饱和）
        4. XY 平面和偏航通道：比例控制
        5. 将所有控制量限幅后发布为速度指令
        """
        # 计算距离上一控制周期的时间间隔 dt
        now = time.time()
        dt = now - self.last_time
        self.last_time = now
        if dt == 0:
            # 防止除零或无效更新
            return

        # 计算各个通道的位置误差
        ex = self.goal_x - self.x   # X 方向误差
        ey = self.goal_y - self.y   # Y 方向误差
        ez = self.goal_z - self.z   # Z 方向误差（高度）
        eyaw = self.wrap(self.goal_yaw - self.yaw)  # 偏航角误差（已归一化到 [-pi, pi]）

        # ----------------------------------------
        #  高度通道 PI 控制
        #  ----------------------------------------
        # 积分项累积：对高度误差进行积分
        self.integral_z += ez * dt
        # 抗积分饱和（anti-windup）：限制积分项在合理范围内
        self.integral_z = max(min(self.integral_z, self.integral_limit), -self.integral_limit)

        # PI 控制律：vz = kp * ez + ki * integral(ez)
        vz = self.kp_z * ez + self.ki_z * self.integral_z
        vz = max(min(vz, 1.0), -1.0)  # 限幅输出到 [-1.0, 1.0]

        # ----------------------------------------
        #  XY 平面 + 偏航角比例控制
        #  ----------------------------------------
        vx = self.kp_xy * ex   # X 方向速度
        vy = self.kp_xy * ey   # Y 方向速度
        wz = self.kp_yaw * eyaw  # 偏航角速度

        # 对 XY 和偏航速度进行限幅
        vx = max(min(vx, 1.0), -1.0)
        vy = max(min(vy, 1.0), -1.0)
        wz = max(min(wz, 1.0), -1.0)

        # 发布速度指令（Twist 消息）
        cmd = Twist()
        cmd.linear.x = vx
        cmd.linear.y = vy
        cmd.linear.z = vz
        cmd.angular.z = wz
        self.cmd_pub.publish(cmd)

    def goto(self, x, y, z, yaw_deg):
        """设置无人机的新目标位姿。

        设置目标位置 (x, y, z) 和目标偏航角，控制器将驱动无人机飞向该点。
        每次设置新目标时，重置高度通道的积分项以防止积分累积过冲。

        参数:
            x (float): 目标 X 坐标（米）
            y (float): 目标 Y 坐标（米）
            z (float): 目标 Z 坐标/高度（米）
            yaw_deg (float): 目标偏航角（度），内部转换为弧度
        """
        self.goal_x = x
        self.goal_y = y
        self.goal_z = z
        self.goal_yaw = math.radians(yaw_deg)  # 角度转弧度
        self.integral_z = 0.0   # 设置新目标时重置积分项，防止过冲
        self.get_logger().info(f"新目标: ({x}, {y}, {z}, 偏航={yaw_deg}°)")

    @staticmethod
    def wrap(a):
        """将角度归一化到 [-pi, pi] 范围内。

        使用 atan2(sin(a), cos(a)) 确保角度始终在合法的周期范围内，
        避免角度误差出现跳变（如从 179° 到 -179° 的跃变）。
        """
        return math.atan2(math.sin(a), math.cos(a))


def main(args=None):
    """主函数：启动无人机位置控制器节点。

    示例任务：设置无人机悬停在 (0, 0, 2.4) 位置，偏航角 0 度。
    节点启动后会持续运行，直到被外部中断。
    """
    rclpy.init(args=args)
    node = DronePositionController()

    # 示例：发送悬停指令，目标位置 (0, 0, 2.4m)，偏航角 0 度
    node.goto(0.0, 0.0, 2.4, 0.0)

    rclpy.spin(node)    # 保持节点运行，等待回调触发
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
