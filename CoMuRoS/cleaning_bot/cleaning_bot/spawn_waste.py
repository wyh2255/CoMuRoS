#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
废物生成模块（用于清洁机器人测试场景）

该模块提供在Ignition Gazebo仿真环境中生成和移除障碍物（红色地面平面）的功能。
主要用于模拟清洁机器人在餐厅环境中遇到障碍物的场景，测试障碍物检测和移除能力。

功能:
  - spawn_ground_plane(): 在仿真环境中生成一个红色半透明地面方块（模拟废物/障碍物）
  - remove_ground_plane(): 移除生成的障碍物（当前未使用，保留备用）
  - main(): 入口函数，生成障碍物并保持20秒后退出
"""

import subprocess
import time

def spawn_ground_plane():
    """
    在Ignition Gazebo仿真中生成一个红色方块（模拟废物/障碍物）

    使用 /world/food_court/create 服务在指定位置创建实体。
    方块位置: (3.5, -3.0, 0.0005)，大小 0.5x0.5x0.001m
    红色材质用于在场景中清晰标识。
    """
    # 定义SDF模型描述
    sdf = '''<?xml version="1.0" ?>
    <sdf version="1.6">
        <model name="small_cube">
            <static>true</static>
            <pose>3.5 -3.0 0.0005 0 0 0</pose>
            <link name="link">
                <visual name="visual">
                    <geometry>
                        <box>
                            <size>0.5 0.5 0.001</size>
                        </box>
                    </geometry>
                    <material>
                        <ambient>1 0 0 1</ambient>
                        <diffuse>1 0 0 1</diffuse>
                        <specular>0.5 0.5 0.5 1</specular>
                        <emissive>0.2 0 0 1</emissive>
                    </material>
                </visual>
            </link>
        </model>
    </sdf>'''

    # 转义SDF字符串以用于命令行参数
    sdf_escaped = sdf.replace('\n', ' ').replace('"', '\\"')

    # 调用Ignition Gazebo的create服务
    cmd = [
        'ign', 'service', '-s', '/world/food_court/create',
        '--reqtype', 'ignition.msgs.EntityFactory',
        '--reptype', 'ignition.msgs.Boolean',
        '--timeout', '1000',
        '--req', f'sdf: "{sdf_escaped}"'
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    print(f"Spawn result: {result.stdout}")
    if result.stderr:
        print(f"Spawn error: {result.stderr}")
    return result.returncode == 0

def remove_ground_plane():
    """
    从仿真环境中移除地面平面（废物）

    调用 /world/food_court/remove 服务移除名为"small_ground_plane"的实体。
    （当前函数保留备用，未在主要流程中使用）
    """
    cmd = [
        'ign', 'service', '-s', '/world/food_court/remove',
        '--reqtype', 'ignition.msgs.Entity',
        '--reptype', 'ignition.msgs.Boolean',
        '--timeout', '1000',
        '--req', 'name: "small_ground_plane", type: 2'
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    print(f"Remove result: {result.stdout}")
    return result.returncode == 0

def main():
    """主函数：在仿真中生成红色障碍物并保持20秒"""
    print("Spawning red ground plane (visual only, no collision)...")
    spawn_ground_plane()
    time.sleep(20)

if __name__ == "__main__":
    main()