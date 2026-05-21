#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
无人机位置控制器客户端模块（Drone Position Controller Client）

该模块实现了一个简单的无人机位置控制服务客户端，用于测试GotoPoseDrone服务。
可以通过调用send_goal方法向无人机位置控制器发送目标位置指令。

主要用于：
  - 测试无人机位置控制器服务的连通性
  - 单独调试无人机的位置控制功能
  - 作为示例代码参考
"""

import rclpy
from rclpy.node import Node
from robot_interface.srv import GotoPoseDrone

NAMESPACE = 'r3'

class DroneGotoClient(Node):
    """无人机位置控制服务客户端节点"""

    def __init__(self):
        """初始化客户端节点，等待服务就绪"""
        super().__init__("drone_goto_client")
        self.declare_parameter('namespace', NAMESPACE)
        self.namespace: str = self.get_parameter('namespace').value

        # 创建服务客户端
        self.cli = self.create_client(GotoPoseDrone, f"/{self.namespace}/goto_pose")

        # 等待服务就绪
        self.get_logger().info(f"Waiting for /{self.namespace}/goto_pose service...")
        self.cli.wait_for_service()
        self.get_logger().info("Service available.")

    def send_goal(self, x, y, z, yaw_deg):
        """
        发送无人机目标位置请求

        参数:
            x, y, z: 目标位置坐标（米）
            yaw_deg: 目标偏航角（度）
        """
        req = GotoPoseDrone.Request()
        req.x = x
        req.y = y
        req.z = z
        req.yaw_deg = yaw_deg

        self.get_logger().info(
            f"Sending drone goto request: x={x}, y={y}, z={z}, yaw={yaw_deg}°"
        )

        # 同步阻塞调用
        future = self.cli.call_async(req)
        rclpy.spin_until_future_complete(self, future)

        if future.result() is not None:
            res = future.result()
            self.get_logger().info(f"[CLIENT] Response: success={res.success}, msg={res.message}")
            return res.success
        else:
            self.get_logger().error("Service call failed.")
            return False


def main(args=None):
    """客户端入口函数：发送测试目标位置"""
    rclpy.init(args=args)

    node = DroneGotoClient()

    # 发送测试目标（可根据需要修改坐标）
    node.send_goal(x=5.0, y=0.0, z=8.0, yaw_deg=90.0)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
