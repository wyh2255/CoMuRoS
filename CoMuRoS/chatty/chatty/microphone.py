#!/usr/bin/env python3
"""
麦克风音频采集节点 —— 从指定音频设备捕获音频流并发布到 ROS2 话题。

使用 sounddevice 库从麦克风采集音频数据，进行下采样处理后发布到
/audio_stream 话题。下采样将设备采样率（如 48kHz）转换为 Whisper
模型所需的 16kHz 采样率，以减少传输和转录的计算开销。
"""
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import sounddevice as sd
import numpy as np
import scipy.signal

class AudioPublisher(Node):
    """麦克风音频发布节点。

    通过定时器周期性地从音频设备采集音频块，进行必要的采样率转换
    （下采样至 16kHz），然后将音频数据以 Float32MultiArray 格式
    发布到 /audio_stream 话题，供语音识别节点使用。
    """
    def __init__(self):
        """初始化麦克风音频发布节点。

        声明 ROS2 参数（设备索引、设备采样率、目标采样率、音频块时长），
        创建 /audio_stream 话题的发布器，并设置定时器周期性采集和发布音频数据。
        """
        super().__init__("microphone")

        # --- 声明 ROS2 参数 ---
        self.declare_parameter("device_index", 5)      # 麦克风设备索引号
        self.declare_parameter("fs_device", 48000)     # 设备原始采样率（Hz）
        self.declare_parameter("fs_target", 16000)     # 目标采样率（Hz，Whisper 推荐 16kHz）
        self.declare_parameter("chunk_duration", 0.5)  # 每个音频块的时长（秒）

        # --- 获取参数值 ---
        self.device_index = self.get_parameter("device_index").value
        self.fs_device = self.get_parameter("fs_device").value
        self.fs_target = self.get_parameter("fs_target").value
        self.chunk_duration = self.get_parameter("chunk_duration").value

        self.pub = self.create_publisher(Float32MultiArray, "/audio_stream", 10)
        self.get_logger().info(
            f"AudioPublisher started: mic={self.device_index}, streaming chunks of {self.chunk_duration}s at {self.fs_device}Hz"
        )
        self.timer = self.create_timer(self.chunk_duration, self.publish_chunk)

    def publish_chunk(self):
        """采集并发布一个音频块。

        从指定麦克风设备录制一个时长为 chunk_duration 的音频块，
        将其从设备采样率下采样至 Whisper 目标采样率（16kHz），
        然后以 Float32MultiArray 格式发布到 /audio_stream 话题。

        下采样使用 scipy.signal.resample 实现，可有效减少数据传输量和 Whisper 的计算开销。
        """
        self.get_logger().debug("Capturing audio chunk...")
        frames = int(self.fs_device * self.chunk_duration)
        audio_chunk = sd.rec(frames, samplerate=self.fs_device, channels=1,
                             dtype="float32", device=self.device_index)
        sd.wait()
        audio_chunk = np.squeeze(audio_chunk)

        # 将音频从设备采样率下采样至目标采样率（16kHz），以适配 Whisper 输入要求
        if self.fs_device != self.fs_target:
            audio_chunk = scipy.signal.resample(
                audio_chunk, int(len(audio_chunk) * self.fs_target / self.fs_device)
            )

        msg = Float32MultiArray()
        msg.data = audio_chunk.tolist()
        self.pub.publish(msg)
        self.get_logger().debug(f"Published {len(msg.data)} samples (downsampled to {self.fs_target}Hz).")

def main(args=None):
    """麦克风音频发布节点主函数。

    初始化 ROS2 节点，创建 AudioPublisher 节点并进入自旋，
    定时采集麦克风音频并发布到 /audio_stream 话题。
    处理键盘中断信号实现优雅关闭。

    Args:
        args: 传递给 rclpy.init() 的命令行参数，默认为 None。
    """
    rclpy.init(args=args)
    node = AudioPublisher()
    try:
        rclpy.spin(node)  # 进入 ROS2 事件循环
    except KeyboardInterrupt:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()