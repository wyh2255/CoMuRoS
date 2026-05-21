# -*- coding: utf-8 -*-
"""
delivery_bot包安装配置脚本（Setup）

配置delivery_bot（配送机器人）的ROS 2 Python包安装信息，
定义数据文件路径和可执行入口点。
"""

from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'delivery_bot'

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
            # 配送机器人LLM控制节点
            'delivery_bot_llm = delivery_bot.delivery_bot_llm:main',
        ],
    },
)
