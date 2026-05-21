"""chatty 包的安装配置。

定义 ROS2 Python 包的元数据、数据文件路径和可执行入口点。
包含聊天系统各节点（GUI、管理器、任务管理器、语音输入输出、
时间发布等）的控制台脚本入口点注册。
"""
from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'chatty'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),

        (os.path.join('share', package_name, 'config'), glob('config/*')),   # 配置文件目录
        (os.path.join('share', package_name, 'launch'), glob('launch/*')),   # 启动文件目录
        (os.path.join('share', package_name, 'data'), glob('data/*')),       # 数据文件目录
        (os.path.join('share', package_name, 'data2'), glob('data2/*')),     # 附加数据文件目录

    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='',
    maintainer_email='',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'chat_gui = chatty.chat_gui:main',           # 聊天 GUI 界面
            # 'chat_gui2 = chatty.chat_gui2:main',       # 备用聊天 GUI（注释）
            'chat_manager = chatty.chat_manager:main',   # 聊天管理器节点
            'task_manager = chatty.task_manager:main',   # 任务管理器节点

            'speak = chatty.speak:main',                 # TTS 语音合成节点
            'time = chatty.time:main',                   # 仿真时间发布节点

            'microphone = chatty.microphone:main',       # 麦克风音频采集节点


            # 测试脚本
            'test_input = chatty.test_input:main',       # 输入测试
            'test_launch = chatty.test_launch:main',     # 启动测试
            'test_launch1 = chatty.test_launch1:main',   # 启动测试 1

        ],
    },
)
