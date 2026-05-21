# -*- coding: utf-8 -*-
"""
yahboom_llm 包安装配置。
注册 LLM 节点和全向移动控制器服务为可执行入口点。
"""
from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'yahboom_llm'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
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
    description='Yahboom Rosmaster X3 LLM 接口包',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        # 控制台入口：ros2 run yahboom_llm <entry_point>
        'console_scripts': [
            'yahboom_llm = yahboom_llm.yahboom_llm:main',
            'holonomic_position_controller_service = yahboom_llm.holonomic_position_controller_service:main',
        ],
    },
)
