#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
无人机位置控制器服务模块（三步PID控制 - 高增益版本）

该模块是三步PID控制器的变体版本，主要区别在于X/Y轴使用更高的比例增益
（kp=4.5 vs 默认版的2.5），以实现更快速的响应。采用与标准版相同的
三步递进式控制策略：先移动到目标位置、再修正偏航角、最后精细微调。

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

class PIDController:
    """通用PID控制器，支持比例-积分-微分控制及抗积分饱和（Anti-windup）"""

    def __init__(self, kp, ki, kd, output_limit=1.0, integral_limit=0.5):
        """
        初始化PID控制器

        参数:
            kp (float): 比例增益
            ki (float): 积分增益
            kd (float): 微分增益
            output_limit (float): 输出限幅值，默认1.0
            integral_limit (float): 积分项限幅值（抗积分饱和），默认0.5
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limit = output_limit
        self.integral_limit = integral_limit

        self.integral = 0.0  # 积分项累积值
        self.prev_error = 0.0  # 前一次偏差值（用于微分计算）

    def reset(self):
        """重置PID状态（清零积分项和上一次偏差）"""
        self.integral = 0.0
        self.prev_error = 0.0

    def compute(self, error, dt):
        """
        计算PID控制输出

        使用位置式PID算法，包含抗积分饱和机制。

        参数:
            error (float): 当前偏差值（目标值 - 当前值）
            dt (float): 时间步长 (s)

        返回:
            float: PID控制输出值（已限幅）
        """
        # 比例项: P = Kp * e(t)
        p_term = self.kp * error

        # 积分项（带抗积分饱和）: I = Ki * integral(e(t)dt)
        self.integral += error * dt
        self.integral = max(min(self.integral, self.integral_limit), -self.integral_limit)
        i_term = self.ki * self.integral

        # 微分项: D = Kd * de(t)/dt
        if dt > 0:
            d_term = self.kd * (error - self.prev_error) / dt
        else:
            d_term = 0.0
        self.prev_error = error

        # 总输出 = P + I + D，带输出限幅
        output = p_term + i_term + d_term
        output = max(min(output, self.output_limit), -self.output_limit)

        return output


class DronePositionController(Node):
    """
    无人机位置控制器（三步PID控制 - 高增益版本）

    与标准三步PID控制器功能相同，但X/Y轴使用更高的比例增益（kp=4.5）
    以实现更快的响应速度。适用于需要快速到达目标的场景。

    三步递进式控制策略:
      步骤1 — 移动到目标位置（X, Y, Z），不控制偏航
      步骤2 — 修正偏航角，同时保持当前位置
      步骤3 — 同时对位置和偏航进行精细微调

    控制策略:
      - X/Y: PID控制 (kp=4.5, ki=0.05, kd=0.8) — 高比例增益快速响应
      - Z: PID控制 (kp=2.5, ki=0.09, kd=0.6)
      - Yaw: PID控制 (kp=1.0, ki=0.1, kd=0.5)
      - 收敛阈值: 位置误差 < 0.05m, 偏航误差 < 0.05rad
    """

    def __init__(self):
        """
        初始化无人机位置控制器

        配置ROS2接口（订阅器、发布器、服务端）、PID控制器参数和状态变量。
        与标准版本不同，X/Y轴使用kp=4.5的高比例增益。
        """
        super().__init__("drone_position_controller")
        self.declare_parameter('namespace', NAMESPACE)
        self.namespace: str = self.get_parameter('namespace').value

        # ========== 回调组 ==========
        # 使用可重入回调组，允许服务回调和订阅回调并发执行
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

        # ========== PID控制器（各轴独立，X/Y使用高增益） ==========
        # X/Y轴: 高比例增益 (kp=4.5)，低积分增益防止超调
        self.pid_x = PIDController(kp=4.5, ki=0.05, kd=0.8, output_limit=1.0)
        self.pid_y = PIDController(kp=4.5, ki=0.05, kd=0.8, output_limit=1.0)
        # Z轴: 中等比例增益，较低积分增益
        self.pid_z = PIDController(kp=2.5, ki=0.09, kd=0.6, output_limit=1.0, integral_limit=0.5)
        # 偏航: 较低比例增益保证稳定性
        self.pid_yaw = PIDController(kp=1.0, ki=0.1, kd=0.5, output_limit=1.0)

        # ========== 收敛阈值 ==========
        self.position_threshold = 0.05  # 位置误差阈值 (m)
        self.yaw_threshold = 0.05       # 偏航角误差阈值 (rad)，约3度

        self.get_logger().info("Three-Step PID Drone Controller Initialized.")

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
    # 阻塞式服务回调 — 三步控制策略
    # -----------------------------------------------------
    def goto_service_callback(self, request, response):
        """
        GotoPoseDrone服务回调（阻塞式三步控制，高增益版本）

        采用三步递进式控制策略：
        1. move_to_position: 移动到目标X/Y/Z位置（不控制偏航）
        2. correct_yaw: 在当前位置修正偏航角
        3. fine_tune_both: 同时对位置和偏航进行精细微调

        由于X/Y使用高比例增益(kp=4.5)，响应速度更快但可能引入超调。
        """
        # 解析服务请求中的目标值
        goal_x = request.x
        goal_y = request.y
        goal_z = request.z
        goal_yaw = math.radians(request.yaw_deg)

        self.cancel_requested = False

        # 重置所有PID控制器状态
        self.pid_x.reset()
        self.pid_y.reset()
        self.pid_z.reset()
        self.pid_yaw.reset()

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

        # ========================================
        # 步骤1: 移动到目标位置 (X, Y, Z)
        # ========================================
        self.get_logger().info("[Drone] Step 1: Moving to position...")
        result = self.move_to_position(goal_x, goal_y, goal_z, timeout=450000.0)

        if not result["success"]:
            self.stop()
            response.success = False
            response.message = result["message"]
            return response

        # ========================================
        # 步骤2: 修正偏航角
        # ========================================
        self.get_logger().info("[Drone] Step 2: Correcting yaw...")
        result = self.correct_yaw(goal_yaw, timeout=10000005.0)

        if not result["success"]:
            self.stop()
            response.success = False
            response.message = result["message"]
            return response

        # ========================================
        # 步骤3: 精细微调位置和偏航
        # ========================================
        self.get_logger().info("[Drone] Step 3: Fine-tuning position and yaw...")
        result = self.fine_tune_both(goal_x, goal_y, goal_z, goal_yaw, timeout=200000.0)

        if not result["success"]:
            self.stop()
            response.success = False
            response.message = result["message"]
            return response

        # 全部步骤成功完成
        self.stop()
        response.success = True
        response.message = "Drone reached goal position and yaw!"
        self.get_logger().info("[Drone] Goal fully reached!")
        return response

    # -----------------------------------------------------
    def move_to_position(self, goal_x, goal_y, goal_z, timeout=45.0):
        """
        步骤1: 使用PID控制移动到目标位置

        在控制循环中直接计算机体坐标系下的速度指令。
        此版本不进行世界坐标系到机体坐标系的旋转变换，
        直接输出PID计算结果作为速度指令。

        参数:
            goal_x (float): 目标X位置 (m)
            goal_y (float): 目标Y位置 (m)
            goal_z (float): 目标Z位置 (m)
            timeout (float): 超时时间 (s)

        返回:
            dict: {"success": bool, "message": str}
        """
        timeout_time = time.time() + timeout
        last_time = time.time()

        while rclpy.ok():
            # 超时检测
            if time.time() > timeout_time:
                return {"success": False, "message": "Position goal timed out!"}

            # 取消检测
            if self.cancel_requested:
                return {"success": False, "message": "Position goal canceled!"}

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

            # PID控制计算速度指令
            vx = self.pid_x.compute(ex, dt)
            vy = self.pid_y.compute(ey, dt)
            vz = self.pid_z.compute(ez, dt)

            # 发布速度指令（步骤1不控制偏航）
            cmd = Twist()
            cmd.linear.x = vx
            cmd.linear.y = vy
            cmd.linear.z = vz
            cmd.angular.z = 0.0
            self.cmd_pub.publish(cmd)

            # 检查是否到达目标位置
            if (abs(ex) < self.position_threshold and
                abs(ey) < self.position_threshold and
                abs(ez) < self.position_threshold):
                self.get_logger().info("[Drone] Position reached!")
                return {"success": True, "message": "Position reached"}

            time.sleep(0.05)

    # -----------------------------------------------------
    def correct_yaw(self, goal_yaw, timeout=15.0):
        """
        步骤2: 修正偏航角，同时保持当前位置

        锁定当前无人机位置作为目标，在保持位置的同时修正偏航角。
        使用PID控制同时输出位置保持指令和偏航修正指令。

        参数:
            goal_yaw (float): 目标偏航角 (rad)
            timeout (float): 超时时间 (s)

        返回:
            dict: {"success": bool, "message": str}
        """
        timeout_time = time.time() + timeout
        last_time = time.time()

        # 记录进入步骤2时的位置作为保持目标
        target_x = self.x
        target_y = self.y
        target_z = self.z

        while rclpy.ok():
            # 超时检测
            if time.time() > timeout_time:
                return {"success": False, "message": "Yaw correction timed out!"}

            # 取消检测
            if self.cancel_requested:
                return {"success": False, "message": "Yaw correction canceled!"}

            # 计算时间步长
            now = time.time()
            dt = now - last_time
            last_time = now
            if dt <= 0:
                dt = 0.01

            # 位置误差（以记录位置为目标，保持当前位置）
            ex = target_x - self.x
            ey = target_y - self.y
            ez = target_z - self.z

            # 偏航角误差（归一化到[-pi, pi]）
            eyaw = self.wrap(goal_yaw - self.yaw)

            # PID控制
            vx = self.pid_x.compute(ex, dt)
            vy = self.pid_y.compute(ey, dt)
            vz = self.pid_z.compute(ez, dt)
            wz = self.pid_yaw.compute(eyaw, dt)

            # 发布速度指令
            cmd = Twist()
            cmd.linear.x = vx
            cmd.linear.y = vy
            cmd.linear.z = vz
            cmd.angular.z = wz
            self.cmd_pub.publish(cmd)

            # 检查偏航角是否到达目标
            if abs(eyaw) < self.yaw_threshold:
                self.get_logger().info("[Drone] Yaw corrected!")
                return {"success": True, "message": "Yaw corrected"}

            time.sleep(0.05)

    # -----------------------------------------------------
    def fine_tune_both(self, goal_x, goal_y, goal_z, goal_yaw, timeout=20.0):
        """
        步骤3: 同时对位置和偏航进行精细微调

        在所有轴上同时进行PID控制，确保位置和偏航角都满足收敛条件。

        参数:
            goal_x (float): 目标X位置 (m)
            goal_y (float): 目标Y位置 (m)
            goal_z (float): 目标Z位置 (m)
            goal_yaw (float): 目标偏航角 (rad)
            timeout (float): 超时时间 (s)

        返回:
            dict: {"success": bool, "message": str}
        """
        timeout_time = time.time() + timeout
        last_time = time.time()

        while rclpy.ok():
            # 超时检测
            if time.time() > timeout_time:
                return {"success": False, "message": "Fine-tuning timed out!"}

            # 取消检测
            if self.cancel_requested:
                return {"success": False, "message": "Fine-tuning canceled!"}

            # 计算时间步长
            now = time.time()
            dt = now - last_time
            last_time = now
            if dt <= 0:
                dt = 0.01

            # 计算所有轴的误差
            ex = goal_x - self.x
            ey = goal_y - self.y
            ez = goal_z - self.z
            eyaw = self.wrap(goal_yaw - self.yaw)

            # PID控制
            vx = self.pid_x.compute(ex, dt)
            vy = self.pid_y.compute(ey, dt)
            vz = self.pid_z.compute(ez, dt)
            wz = self.pid_yaw.compute(eyaw, dt)

            # 发布速度指令
            cmd = Twist()
            cmd.linear.x = vx
            cmd.linear.y = vy
            cmd.linear.z = vz
            cmd.angular.z = wz
            self.cmd_pub.publish(cmd)

            # 检查是否所有轴都满足收敛条件
            if (abs(ex) < self.position_threshold and
                abs(ey) < self.position_threshold and
                abs(ez) < self.position_threshold and
                abs(eyaw) < self.yaw_threshold):
                self.get_logger().info("[Drone] Fine-tuning complete!")
                return {"success": True, "message": "Fine-tuning complete"}

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

    初始化无人机PID位置控制器节点（高增益版本）并开始ROS2事件循环。
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
