# -*- coding: utf-8 -*-
"""
机器人LLM节点启动文件（Robot LLMs Launch File）

该启动文件用于同时启动多个机器人的LLM控制节点：
  - cleaning_bot_llm_node：清洁机器人LLM节点
  - delivery_bot_llm_node：配送机器人LLM节点
  - drone_llm_node：无人机LLM节点

每个节点通过参数指定机器人名称，以便区分不同的机器人实例。
"""

from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    """生成启动描述，包含三个机器人LLM节点"""

    # 清洁机器人LLM节点
    cleaning_robot_node = Node(
        package='cleaning_bot',
        executable='cleaning_bot_llm',
        name='cleaning_bot_llm_node',
        parameters=[{
            'robot_name': 'cleaning_bot',
        }],
        output='screen',
    )

    # 配送机器人LLM节点
    delivery_bot_node = Node(
        package='delivery_bot',
        executable='delivery_bot_llm',
        name='delivery_bot_llm_node',
        parameters=[{
            'robot_name': 'delivery_bot',
        }],
        output='screen',
    )

    # 无人机LLM节点
    drone_node = Node(
        package='drone',
        executable='drone_llm',
        name='drone_llm_node',
        parameters=[{
            'robot_name': 'drone',
        }],
        output='screen',
    )

    return LaunchDescription([
        cleaning_robot_node,
        delivery_bot_node,
        drone_node,
    ])