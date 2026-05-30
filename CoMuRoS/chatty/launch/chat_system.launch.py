"""chatty 系统启动文件。

定义 chatty 包中所有 ROS2 节点的启动配置，包括：
聊天界面、聊天管理器、任务管理器、麦克风输入、语音输出和
时间发布等节点。支持通过启动参数控制音频输入输出的启用状态。
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """生成 chatty 系统的完整启动描述。

    配置以下节点及其参数：
    - chat_gui: 聊天 GUI 界面
    - chat_manager: 聊天管理器
    - task_manager: 任务管理器（支持 model 和 config_file 参数）
    - microphone: 麦克风音频采集（条件启动）
    - tts_speaker: 语音合成输出（条件启动）
    - whisper_listener: 语音识别输入（条件启动）
    - time_publisher: 仿真时间发布

    可通过 launch 参数控制音频输入/输出的启用：
    - enable_audio_input: 是否启用麦克风和语音识别
    - enable_audio_output: 是否启用语音合成输出

    Returns:
        LaunchDescription: ROS2 启动描述对象，包含所有声明参数和节点。
    """

    # -------------------- 参数替换变量 --------------------

    model       = LaunchConfiguration('model')            # 模型编号
    config_file = LaunchConfiguration('config_file')      # 配置文件名称
    enable_audio_input  = LaunchConfiguration('enable_audio_input')   # 音频输入启用标志
    enable_audio_output = LaunchConfiguration('enable_audio_output')  # 音频输出启用标志
    coordinator_url = LaunchConfiguration('coordinator_url')

    # -------------------- 启动参数声明 --------------------

    model_arg = DeclareLaunchArgument(
        'model',default_value='10', description='模型编号'
    )

    config_file_arg = DeclareLaunchArgument(
        'config_file', default_value='robot_config_roscon_2025', description='配置文件名称'
    )

    enable_audio_input_arg = DeclareLaunchArgument(
        'enable_audio_input',
        default_value='false',
        description='启用麦克风输入和语音识别'
    )

    enable_audio_output_arg = DeclareLaunchArgument(
        'enable_audio_output',
        default_value='false',
        description='启用文本转语音输出'
    )

    coordinator_url_arg = DeclareLaunchArgument(
        'coordinator_url',
        default_value='http://localhost:8080',
        description='A2A Coordinator URL',
    )

    # -------------------- 节点定义 --------------------

    chat_interface_node = Node(
        package='chatty',
        executable='chat_gui',
        name='chat_gui',
        output='screen',
        parameters=[{
            'coordinator_url': LaunchConfiguration('coordinator_url'),
            'config_file': LaunchConfiguration('config_file'),
        }],
    )

    chat_manager_node = Node(
        package='chatty',
        executable='chat_manager',
        name='chat_manager',
        # parameters=[{
        #     'config_file': config_file
        # }],
        output='screen',
    )

    task_manager_node = Node(
        package='chatty',
        executable='task_manager',
        name='task_manager',
        env_vars={'USE_A2A_COORDINATOR': '1'},
        parameters=[{
            'model': model,
            'config_file': config_file,
        }],
        output='screen',
    )

    microphone_node = Node(              # 麦克风采集节点
        package='chatty',
        executable='microphone',
        name='microphone',
        output='screen',
        condition=IfCondition(enable_audio_input)  # 仅在启用音频输入时启动
    )

    speaking_node = Node(                # 语音合成输出节点
        package='chatty',
        executable='speak',
        name='tts_speaker',
        output='screen',
        condition=IfCondition(enable_audio_output)  # 仅在启用音频输出时启动
    )

    listening_node = Node(               # 语音识别输入节点
        package='chatty',
        executable='listen',
        name='whisper_listener',
        output='screen',
        condition=IfCondition(enable_audio_input)  # 仅在启用音频输入时启动
    )

    time_pub_node = Node(                # 仿真时间发布节点
        package='chatty',
        executable='time',
        name='time_publisher',
        output='screen'
    )

    # -------------------- 返回完整的启动描述 --------------------

    return LaunchDescription([
        model_arg,
        config_file_arg,
        enable_audio_input_arg,
        enable_audio_output_arg,
        coordinator_url_arg,

        chat_interface_node,
        chat_manager_node,
        task_manager_node,

        microphone_node,
        speaking_node,
        listening_node,

        time_pub_node
    ])

