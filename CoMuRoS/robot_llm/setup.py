# -*- coding: utf-8 -*-
"""
robot_llm包安装配置脚本（Setup）

配置cleaning_bot（清洁机器人）的ROS 2 Python包安装信息，
定义数据文件路径和可执行入口点。
"""

from setuptools import find_packages, setup
from glob   import glob
import os

package_name = 'robot_llm'

setup(
    name=package_name,
    version='0.0.0',
    # 自动发现包目录（排除test目录）
    packages=find_packages(exclude=['test']),
    # 数据文件：资源索引、package.xml、启动文件、数据文件
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*')),
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
            'robot_llm_node = robot_llm.robot_llm:main',
            'mobile_robot = robot_llm.mobile_robot:main',
            'manipulator_robot = robot_llm.manipulator_robot:main',
        ],
    },
)
