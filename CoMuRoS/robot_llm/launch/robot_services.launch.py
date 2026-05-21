# -*- coding: utf-8 -*-
"""
机器人位置控制服务启动文件（Robot Services Launch File）

该启动文件用于启动各机器人的位置控制器服务节点：
  - cleaning_bot_position_controller：清洁机器人位置控制器（命名空间：r1）
  - deliver_bot_position_controller：配送机器人位置控制器（命名空间：r2）
  - drone_position_controller：无人机位置控制器（命名空间：r3）

每个控制器通过namespace参数区分，用于接收位置指令并驱动机器人移动到目标位姿。
"""

from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    """生成启动描述，包含三个机器人位置控制器服务节点"""

    # 全向机器人1号（清洁机器人）位置控制器
    holonomic_robot1 = Node(
        package='cleaning_bot',
        executable='holonomic_position_controller_service',
        name='cleaning_bot_position_controller',
        parameters=[{
            'namespace': 'r1',
        }],
        output='screen',
    )

    # 全向机器人2号（配送机器人）位置控制器
    holonomic_robot2 = Node(
        package='cleaning_bot',
        executable='holonomic_position_controller_service',
        name='deliver_bot_position_controller',
        parameters=[{
            'namespace': 'r2',
        }],
        output='screen',
    )

    # 无人机位置控制器
    drone_1 = Node(
        package='drone',
        executable='drone_position_controller_service',
        name='drone_position_controller',
        parameters=[{
            'namespace': 'r3',
        }],
        output='screen',
    )

    return LaunchDescription([
        holonomic_robot1,
        holonomic_robot2,
        drone_1,
    ])