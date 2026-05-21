#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
无人机位置控制客户端模块

该模块实现了无人机（四旋翼）的GotoPose服务客户端，用于向无人机位置控制器服务
发送目标位置请求。支持三维空间中的位置控制（x, y, z）和偏航角控制。

ROS2接口:
  - 服务客户端: /{namespace}/goto_pose (GotoPoseDrone) — 位置控制服务

用法:
  直接运行脚本即可发送预设的目标位置（默认: x=5.0, y=-1.0, z=5.3, yaw=90°）
"""

import rclpy
from rclpy.node import Node
from robot_interface.srv import GotoPoseDrone

# 无人机机器人命名空间
NAMESPACE = 'r3'

class DroneGotoClient(Node):
    """
    无人机位置控制服务客户端

    作为ROS2节点，创建GotoPoseDrone服务的客户端，通过同步阻塞方式
    向无人机位置控制器发送目标位姿请求并等待结果。
    """

    def __init__(self):
        """
        初始化无人机客户端节点

        创建GotoPoseDrone服务客户端并等待服务端就绪。
        """
        super().__init__("drone_goto_client")
        self.declare_parameter('namespace', NAMESPACE)
        self.namespace: str = self.get_parameter('namespace').value

        # 创建服务客户端，连接到指定命名空间下的goto_pose服务
        self.cli = self.create_client(GotoPoseDrone, f"/{self.namespace}/goto_pose")

        # 等待服务端就绪（阻塞等待）
        self.get_logger().info(f"Waiting for /{self.namespace}/goto_pose service...")
        self.cli.wait_for_service()
        self.get_logger().info("Service available.")

    def send_goal(self, x, y, z, yaw_deg):
        """
        发送目标位置请求（同步阻塞）

        向位置控制器发送三维目标位置和偏航角，等待服务执行完成并返回结果。

        参数:
            x (float): 目标X位置 (m)
            y (float): 目标Y位置 (m)
            z (float): 目标Z位置/高度 (m)
            yaw_deg (float): 目标偏航角 (度)

        返回:
            bool: 是否成功到达目标
        """
        # 构建GotoPoseDrone服务请求
        req = GotoPoseDrone.Request()
        req.x = x
        req.y = y
        req.z = z
        req.yaw_deg = yaw_deg

        self.get_logger().info(
            f"Sending drone goto request: x={x}, y={y}, z={z}, yaw={yaw_deg}°"
        )

        # 异步调用服务并通过spin_until_future_complete实现同步等待
        future = self.cli.call_async(req)
        rclpy.spin_until_future_complete(self, future)

        # 处理服务响应结果
        if future.result() is not None:
            res = future.result()
            self.get_logger().info(f"[CLIENT] Response: success={res.success}, msg={res.message}")
            return res.success
        else:
            self.get_logger().error("Service call failed.")
            return False


def main(args=None):
    """
    主入口函数

    初始化无人机客户端节点，发送预设目标位置并退出。
    默认目标: x=5.0, y=-1.0, z=5.3 (高度), yaw=90°
    """
    rclpy.init(args=args)

    node = DroneGotoClient()

    # 可在此修改目标位置参数
    # node.send_goal(x=5.0, y=-1.0, z=5.3, yaw_deg=90.0)
    node.send_goal(x=5.0, y=-1.0, z=5.3, yaw_deg=90.0)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()


'''

备用：Ignition Gazebo直接设置无人机位姿的命令行服务调用

该命令可直接在仿真环境中设置无人机r3的位姿，无需经过位置控制器：
  - name: "r3" — 无人机模型名称
  - position: 目标三维坐标 (x, y, z)
  - orientation: 四元数表示的朝向 (w, x, y, z)，此处对应90度偏航

ign service -s /world/food_court/set_pose \
  --reqtype ignition.msgs.Pose \
  --reptype ignition.msgs.Boolean \
  --timeout 5000 \
  --req 'name: "r3" position: { x: 5.0 y: -1.0 z: 5.3 } orientation: { x: 0.0 y: 0.0 z: 0.7071 w: 0.7071 }'


'''
