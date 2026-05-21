#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多机器人仿真启动文件（Multi-Robot Launch File）

该文件用于在Ignition Gazebo仿真环境中启动多个机器人：
  - robot3（无人机）：x3_uav，5秒后延迟启动，前缀r3，位置(-1.0, 1.0)
  - robot2（全向机器人）：rosmaster_x3，15秒后延迟启动，前缀r2，位置(0.0, 0.0)
  - robot1（全向机器人）：rosmaster_x3，25秒后延迟启动，前缀r1，位置(11.0, 0.0)

使用TimerAction实现逐步启动，避免同时加载引起的资源竞争。
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    TimerAction
)
from launch.substitutions import PathJoinSubstitution
from launch.conditions import IfCondition
from launch.substitutions import AndSubstitution, NotSubstitution


def generate_launch_description():
    """生成多机器人仿真启动描述"""

    # 获取各功能包路径
    pkg_shared            = get_package_share_directory('multi_robot')
    pkg_yahboom_gazebo    = get_package_share_directory('yahboom_rosmaster_gazebo')
    pkg_x3_uav_gazebo     = get_package_share_directory('x3_uav_ignition')
    pkg_ros_gz_sim        = get_package_share_directory('ros_gz_sim')

    # 启动参数声明
    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time', default_value='true', description='Use simulation time'
    )
    declare_use_ignition = DeclareLaunchArgument(
        'use_ignition', default_value='true', description='Enable Ignition Gazebo plugins'
    )
    decalre_use_headless_arg = DeclareLaunchArgument(
        'headless', default_value='false', description='use igntion headless'
    )
    declare_single_rviz_cmd = DeclareLaunchArgument(
        'use_single_rviz', default_value='false', description='use one rviz for all the robot'
    )
    decalre_multi_rviz_cmd = DeclareLaunchArgument(
        'use_multi_rviz', default_value='false', description='use one rviz for each robot'
    )

    # 启动配置变量
    use_sim_time     = LaunchConfiguration('use_sim_time')
    use_ignition     = LaunchConfiguration('use_ignition')
    headless         = LaunchConfiguration('headless')
    use_single_rviz  = LaunchConfiguration('use_single_rviz')
    use_multi_rviz   = LaunchConfiguration('use_multi_rviz')

    # Gazebo世界文件路径
    world_file = os.path.join(
        pkg_shared,
        'worlds',
        'food_court.sdf'
    )

    # 启动Gazebo服务器（无头模式）
    start_gz_server_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_ros_gz_sim, 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={
            'gz_args': [ '-r -s -v2 ', world_file ],
            'on_exit_shutdown': 'true'
        }.items(),
        condition=IfCondition(use_ignition)
    )

    # 启动Gazebo客户端（GUI，非无头模式时）
    start_gz_client_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_ros_gz_sim, 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={
            'gz_args': '-g -v2'
        }.items(),
        condition=IfCondition(
            AndSubstitution(
                NotSubstitution(headless),
                use_ignition
            )
        )
    )

    # 时钟和TF桥接节点
    igntion_bridge_cmd = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='clock_tf_bridge',
        arguments=[
            '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
            '/tf@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V',
        ],
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen',
    )

    # 机器人启动文件路径
    yahboom_launch = PathJoinSubstitution([
        pkg_yahboom_gazebo, 'launch', 'yahboom_robot.launch.py'
    ])

    x3_uav_robot = PathJoinSubstitution([
        pkg_x3_uav_gazebo, 'launch', 'x3_uav_robot.launch.py'
    ])

    # 机器人3：无人机（5秒后启动）
    robot3_launch = TimerAction(
        period=5.0,
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(x3_uav_robot),
                launch_arguments={
                    'use_sim_time': use_sim_time,
                    'robot_name': 'x3_uav',
                    'prefix': 'r3',
                    'use_ignition': use_ignition,
                    'use_plugin': 'True',
                    'use_ros2_control': 'false',
                    'use_mock_hardware': 'false',
                    'spawn_x': '-1.0',
                    'spawn_y': '1.0',
                    'spawn_z': '0.05',
                    'spawn_roll': '0.0',
                    'spawn_pitch': '0.0',
                    'spawn_yaw': '0.0',
                    'use_rviz': use_multi_rviz,
                }.items()
            )
        ]
    )

    # 机器人2：全向机器人（15秒后启动）
    robot2_launch = TimerAction(
        period=15.0,
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(yahboom_launch),
                launch_arguments={
                    'use_sim_time': use_sim_time,
                    'robot_name': 'rosmaster_x3',
                    'prefix': 'r2',
                    'use_ignition': use_ignition,
                    'use_plugin': 'True',
                    'use_ros2_control': 'False',
                    'use_mock_hardware': 'False',
                    'spawn_x': '0.0',
                    'spawn_y': '0.0',
                    'spawn_z': '0.05',
                    'spawn_roll': '0.0',
                    'spawn_pitch': '0.0',
                    'spawn_yaw': '0.0',
                    'use_rviz': use_multi_rviz,
                }.items()
            )
        ]
    )

    # 机器人1：全向机器人（25秒后启动，放置在远端位置）
    robot1_launch = TimerAction(
        period=25.0,
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(yahboom_launch),
                launch_arguments={
                    'use_sim_time': use_sim_time,
                    'robot_name': 'rosmaster_x3',
                    'prefix': 'r1',
                    'use_ignition': use_ignition,
                    'use_plugin': 'True',
                    'use_ros2_control': 'False',
                    'use_mock_hardware': 'False',
                    'spawn_x': '11.0',
                    'spawn_y': '0.0',
                    'spawn_z': '0.05',
                    'spawn_roll': '0.0',
                    'spawn_pitch': '0.0',
                    'spawn_yaw': '0.0',
                    'use_rviz': use_multi_rviz,
                }.items()
            )
        ]
    )

    # 单窗口RViz节点（显示所有机器人）
    single_rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='single_rviz',
        arguments=['-d', os.path.join(pkg_shared, 'rviz', 'multi_robot.rviz')],
        parameters=[{'use_sim_time': use_sim_time}],
        condition=IfCondition(use_single_rviz),
        output='screen'
    )

    return LaunchDescription([
        declare_use_sim_time,
        declare_use_ignition,
        decalre_use_headless_arg,
        decalre_multi_rviz_cmd,
        declare_single_rviz_cmd,

        start_gz_server_cmd,
        start_gz_client_cmd,

        igntion_bridge_cmd,

        robot3_launch,
        robot2_launch,
        robot1_launch,

        single_rviz_node
    ])