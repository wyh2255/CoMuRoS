# -*- coding: utf-8 -*-
"""
drone包安装配置脚本（Setup）

配置drone（无人机）的ROS 2 Python包安装信息，
定义数据文件路径和可执行入口点（LLM节点、位置控制器服务/客户端）。
"""

from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'drone'

setup(
    name=package_name,
    version='0.0.0',
    # 自动发现包目录（排除test目录）
    packages=find_packages(exclude=['test']),
    # 数据文件：资源索引、package.xml、数据文件
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'data'), glob('data/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='name',
    maintainer_email='name@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    # 可执行入口点：注册ROS 2节点
    entry_points={
        'console_scripts': [
            # 无人机LLM控制节点
            'drone_llm = drone.drone_llm:main',
            # 无人机位置控制器服务（PID控制）
            'drone_position_controller_service = drone.drone_position_controller_service:main',
            # 无人机位置控制器测试客户端
            'drone_position_controller_client = drone.drone_position_controller_client:main',
        ],
    },
)
