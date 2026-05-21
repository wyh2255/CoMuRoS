#!/usr/bin/env python3
"""
语音转文本节点 —— 使用 OpenAI Whisper 模型将音频流实时转录为文本。

订阅 /audio_stream 话题获取麦克风音频数据，订阅 /switch_state 话题获取
开关状态，在开关按下时缓冲音频，松开时进行转录并将结果发布到 /chat/input 话题。
同时通过 /face_mode 话题控制机器人面部表情状态。
"""
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, String, Int32
import numpy as np
import torch
import whisper
import threading

class WhisperSubscriber(Node):
    """Whisper 语音识别节点。

    负责从音频话题接收音频流，在开关按下时缓冲音频数据，在开关松开时
    调用 Whisper 模型进行离线转录，将识别结果发布到聊天输入话题，
    并在转录过程中控制机器人面部表情模式。
    """
    def __init__(self):
        """初始化 Whisper 语音识别节点。

        声明 ROS2 参数（模型名称、是否使用 CUDA），订阅音频流和开关状态话题，
        创建转录结果发布器和面部表情发布器。加载 Whisper 模型并初始化音频缓冲区和线程锁。
        """
        super().__init__("whisper_subscriber")

        # --- 声明 ROS2 参数 ---
        self.declare_parameter("model_name", "medium")  # Whisper 模型名称（tiny/base/small/medium/large）
        self.declare_parameter("use_cuda", True)        # 是否使用 GPU 加速

        model_name = self.get_parameter("model_name").value
        use_cuda = self.get_parameter("use_cuda").value

        # --- 创建订阅器和发布器 ---
        self.audio_sub = self.create_subscription(Float32MultiArray, "/audio_stream",self.audio_callback, 10)
        self.switch_sub = self.create_subscription(String, "/switch_state",self.switch_callback, 10)

        self.pub = self.create_publisher(String, "/chat/input", 10)         # 转录结果发布话题
        self.face_mode_pub = self.create_publisher(Int32, "/face_mode", 10) # 面部表情控制话题

        # --- 加载 Whisper 模型 ---
        self.device = "cuda" if (torch.cuda.is_available() and use_cuda) else "cpu"
        self.model = whisper.load_model(model_name, device=self.device)
        self.get_logger().info(f"Loaded Whisper model '{model_name}' on {self.device}")
        # --- 音频缓冲区和状态变量 ---
        self.audio_buffer = []               # 音频数据缓冲区
        self.buffering = False               # 是否正在缓冲音频
        self.switch_state = None              # 当前开关状态
        self.prev_switch_state = None         # 上一次开关状态（用于边沿检测）
        self.buffer_lock = threading.Lock()   # 缓冲区线程锁，确保线程安全
        self.get_logger().info("Ready: Switch to PRESSED to start buffering, RELEASED to stop and transcribe.")

    def switch_callback(self, msg):
        """开关状态回调函数。

        接收 /switch_state 话题的开关状态消息，更新本地开关状态变量。
        状态值通常为 "PRESSED"（按下）或 "RELEASED"（松开）。

        Args:
            msg: String 类型的 ROS2 消息，包含开关状态文本。
        """
        self.switch_state = msg.data

    def audio_callback(self, msg):
        """音频数据回调函数。

        接收麦克风发布的音频数据。当节点处于缓冲状态时，
        将音频块追加到缓冲区中。使用线程锁确保并发安全。

        Args:
            msg: Float32MultiArray 类型的 ROS2 消息，包含音频采样数据。
        """
        with self.buffer_lock:
            if self.buffering:
                audio_chunk = np.array(msg.data, dtype=np.float32)
                self.audio_buffer.append(audio_chunk)
                self.get_logger().debug(f"Buffered chunk: {len(audio_chunk)} samples, shape: {audio_chunk.shape}")

    def transcribe_and_publish(self):
        """转录缓冲区音频并发布结果。

        从缓冲区复制并清空音频数据，检查音频长度是否足够（至少 1 秒 @ 16kHz），
        调用 Whisper 模型进行转录。将转录文本以 "human|" 前缀格式发布到 /chat/input 话题。
        该函数通常在独立线程中执行，以避免阻塞主循环。
        """
        with self.buffer_lock:
            local_buffer = self.audio_buffer.copy()  # 线程安全的缓冲区复制
            self.audio_buffer = []                     # 清空缓冲区
        # 检查音频长度是否至少为 1 秒（@ 16kHz 采样率需 16000 个采样点）
        if local_buffer and len(local_buffer) * len(local_buffer[0]) > 16000:
            audio = np.concatenate(local_buffer)       # 合并所有音频块
            self.get_logger().info(f"Running transcription on buffered audio ({len(audio)} samples)...")
            # 调用 Whisper 模型进行转录，指定语言为英语
            result = self.model.transcribe(audio, fp16=(self.device == "cuda"), language="en")
            text = result["text"].strip()
            if text:
                msg_out = String()
                msg_out.data = f"human|{text}"  # 在转录文本前添加 "human|" 前缀标记
                self.pub.publish(msg_out)
                self.get_logger().info(f"Transcript: {text}")
            else:
                self.get_logger().warn("Whisper returned empty text, skipping publish. (Expected for non-English input)")
            self.get_logger().info("Transcription complete. Ready for next: Switch to PRESSED to start buffering.")
        else:
            self.get_logger().warn(f"No sufficient audio buffered ({len(local_buffer)} chunks).")

def main(args=None):
    """Whisper 语音识别节点主函数。

    手动实现 ROS2 事件循环，通过轮询开关状态实现音频缓冲控制。
    使用防抖（debounce）机制避免开关抖动导致的误触发，并通过独立线程
    执行转录操作以避免阻塞音频缓冲。

    Args:
        args: 传递给 rclpy.init() 的命令行参数，默认为 None。
    """
    if not rclpy.ok():  # 仅在 rclpy 未初始化时进行初始化
        rclpy.init(args=args)
    node = WhisperSubscriber()
    last_state_change = 0.0            # 上次状态改变时间戳
    debounce_duration = 0.5            # 防抖时间窗口（秒），防止开关抖动
    try:
        while rclpy.ok():
            current_switch = node.switch_state
            # 获取当前时间（秒 + 纳秒转换为秒）
            current_time = node.get_clock().now().to_msg().sec + node.get_clock().now().to_msg().nanosec * 1e-9
            # 开关按下时持续发布面部表情为"正在说话"状态（1）
            if current_switch == "PRESSED":
                face_mode_msg = Int32()
                face_mode_msg.data = 1
                node.face_mode_pub.publish(face_mode_msg)
            # 检测开关从"松开"到"按下"的上升沿，带防抖
            if (current_switch == "PRESSED" and node.prev_switch_state != "PRESSED" and
                not node.buffering and (current_time - last_state_change) > debounce_duration):
                node.buffering = True
                with node.buffer_lock:
                    node.audio_buffer = []  # 清空旧缓冲区，准备新录音
                node.get_logger().info("Started buffering audio...")
                last_state_change = current_time
            # 检测开关从"按下"到"松开"的下降沿，带防抖
            if (current_switch == "RELEASED" and node.prev_switch_state == "PRESSED" and
                node.buffering and (current_time - last_state_change) > debounce_duration):
                node.buffering = False
                # 开关松开时发布面部表情为"安静"状态（0）
                face_mode_msg = Int32()
                face_mode_msg.data = 0
                node.face_mode_pub.publish(face_mode_msg)
                # 在新线程中启动转录，避免阻塞音频缓冲
                threading.Thread(target=node.transcribe_and_publish).start()
                last_state_change = current_time
            node.prev_switch_state = current_switch  # 保存当前状态供下次比较
            rclpy.spin_once(node, timeout_sec=0.1)   # 非阻塞轮询，等待 0.1 秒
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down node...")
    finally:
        node.destroy_node()
        if rclpy.ok():  # 仅在上下文仍活跃时关闭 rclpy
            rclpy.shutdown()

if __name__ == "__main__":
    main()