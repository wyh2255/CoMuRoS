#!/usr/bin/env python3
"""
文本转语音节点 —— 使用 Edge-TTS 将文本消息转换为语音并播放。

订阅 /chat/output 话题获取聊天输出文本，解析"Task Manager:"前缀的指令，
过滤不相关内容，通过 Edge-TTS 引擎进行语音合成并使用 aplay 播放。
使用异步队列实现顺序播放，避免并发语音重叠。
"""
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Int32
import edge_tts
from pydub import AudioSegment
import subprocess
import tempfile
import os
import re
import asyncio
import threading
import random
from queue import Queue, Empty

class TTSSpeaker(Node):
    """TTS 语音合成节点。

    监听 /chat/output 话题，提取任务管理器相关消息，过滤掉需要跳过或
    不需要朗读的内容，使用 Edge-TTS（微软 Azure 认知服务）进行语音合成。
    通过工作线程和异步队列实现顺序播放，并支持关键词匹配后的随机提示语播报。
    """
    def __init__(self):
        """初始化 TTS 语音合成节点。

        声明 ROS2 参数（语音名称、语速），订阅 /chat/output 话题。
        编译正则表达式用于快速提取任务管理器消息，定义关键词过滤规则
        和随机提示语列表。创建异步事件循环和工作线程，实现队列化顺序语音合成。
        """
        super().__init__('tts_speaker')

        # --- 声明 ROS2 参数 ---
        self.declare_parameter('voice_name', 'en-CA-ClaraNeural')  # TTS 语音名称
        self.declare_parameter('speech_rate', '-10%')              # 语速调整（负值减慢）

        self.sub = self.create_subscription(String, '/chat/output', self.chat_output_callback, 10)
        # self.face_pub = self.create_publisher(Int32, '/face_mode', 10)  # 面部表情发布器（暂未启用）

        # --- 预编译正则表达式，提高提取速度 ---
        self.task_regex = re.compile(r'Task Manager:\s*(.+)', re.IGNORECASE | re.DOTALL)


        # --- 过滤关键词和备用提示语句 ---
        self.skip_keywords = ["independent task", "plan", "insdependent"]             # 匹配到这些关键词时跳过原文
        self.never_speak_words = ["all the helps are used", "multi-robot task completed successfully","excellent work"]  # 完全静默的词语

        self.attention_list = [   # 跳过原文时播放的随机提示语列表
            # "Let me give you a hint!",
            # "Here's a small clue check this out!",
            # "Here's a little hint for you!",
            # "A small hint coming your way!",
            "Hey, You may try this!",

        ]


        # --- TTS 队列和工作线程 ---
        self.tts_queue = Queue()                                  # TTS 任务队列，保证顺序播放
        self.stop_event = threading.Event()                       # 停止事件标志
        self.loop = asyncio.new_event_loop()                      # 为工作线程创建独立异步事件循环
        self.worker_thread = threading.Thread(target=self.tts_worker, daemon=True)  # 后台工作线程（守护线程）
        self.worker_thread.start()

        self.get_logger().info("TTSSpeaker active: Listening for Task Manager messages on /chat/output")

    def chat_output_callback(self, msg: String):
        """聊天输出回调函数。

        接收 /chat/output 话题的消息，使用正则表达式提取"Task Manager:"后的内容，
        进行文本清洗和关键词过滤，决定是否将文本加入 TTS 播放队列。
        若匹配到跳过关键词，则播放随机提示语代替原文。

        Args:
            msg: String 类型的 ROS2 消息，包含聊天输出文本。
        """
        def sanitize_for_tts(text: str) -> str:
            """清洗文本以适配 TTS 引擎。

            移除可能干扰文本转语音处理的特殊字符（换行符、制表符、
            括号、星号等），将连续空白归一化为单个空格。

            Args:
                text: 原始输入文本。

            Returns:
                清洗后的纯文本，适合 TTS 引擎处理。
            """
            import re
            # 替换所有换行符、制表符和特殊符号为单个空格
            text = re.sub(r'[\n\r\f\v\t*(){}[\]<>]+', ' ', text)
            # 将所有连续空白（包括多个空格）归一化为单个空格
            text = re.sub(r'\s+', ' ', text)
            # 去除首尾空格
            return text.strip()

        # 使用正则提取"Task Manager:"后的文本内容
        match = self.task_regex.search(msg.data)
        if match:
            text = match.group(1).strip()
            lower_text = text.lower()

            sanitized_text = sanitize_for_tts(lower_text)
            self.get_logger().info(f"Sanitized text: {repr(sanitized_text)}")  # 调试日志，显示清洗后文本的精确表示

            # 如果文本包含"永不朗读"的词语，直接静默忽略
            if any(nsw in sanitized_text for nsw in self.never_speak_words):
                return

            # 如果文本包含跳过关键词，播放随机提示语代替原文
            if any(kw in sanitized_text for kw in self.skip_keywords):
                self.get_logger().info(f"Skipping due to keyword match: {sanitized_text}")
                attention_text = random.choice(self.attention_list)
                self.tts_queue.put(attention_text)
                self.get_logger().info(f"Triggered attention phrase: {attention_text}")
                return

            # 正常文本直接加入 TTS 播放队列
            self.tts_queue.put(sanitized_text)
            self.get_logger().info(f"Queued for TTS: {sanitized_text[:150]}{'...' if len(sanitized_text) > 150 else ''}")

    def tts_worker(self):
        """TTS 后台工作线程入口。

        为当前线程设置独立的事件循环，然后运行异步 TTS 主循环。
        这样可以将 TTS 的异步操作与 ROS2 主线程隔离开。
        """
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self._tts_loop())

    async def _tts_loop(self):
        """异步 TTS 主循环。

        持续从队列中获取待合成的文本，调用 speak_text 进行语音合成和播放。
        使用超时机制实现非阻塞队列读取；队列为空时短暂休眠避免忙等。

        Raises:
            Empty: 队列为空时的预期异常，用于触发休眠等待。
        """
        while not self.stop_event.is_set():
            try:
                # 从队列获取文本（超时 0.2 秒），通过 asyncio.to_thread 避免阻塞事件循环
                text = await asyncio.to_thread(self.tts_queue.get, True, 0.2)
                await self.speak_text(text)
                self.tts_queue.task_done()
            except Empty:
                await asyncio.sleep(0.05)  # 队列为空时短暂休眠
            except Exception as e:
                self.get_logger().error(f"TTS loop error: {e}")

    async def speak_text(self, text: str):
        """将文本合成为语音并播放（异步实现）。

        使用 Edge-TTS 生成 MP3 音频文件，通过 pydub 转换为 WAV 格式，
        最后调用系统 aplay 命令播放音频。播放完成后清理临时文件。

        Args:
            text: 需要朗读的文本内容。
        """
        voice = self.get_parameter('voice_name').value
        speech_rate = self.get_parameter('speech_rate').value

        self.publish_face_mode(-1)  # 播放前设置面部表情为"正在说话"

        mp3_path, wav_path = None, None
        try:
            # 创建临时 MP3 文件
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as mp3_file:
                mp3_path = mp3_file.name
            wav_path = mp3_path.replace('.mp3', '.wav')

            # 使用 Edge-TTS 生成语音（微软 Azure TTS 服务）
            # await edge_tts.Communicate(text, voice).save(mp3_path)
            communicate = edge_tts.Communicate(text, voice, rate=speech_rate)
            await communicate.save(mp3_path)

            # 将 MP3 转换为 WAV 格式并用 aplay 播放
            AudioSegment.from_mp3(mp3_path).export(wav_path, format='wav')
            subprocess.run(["aplay", "-D", "default", "--quiet", wav_path], check=True)

        except Exception as e:
            self.get_logger().error(f"TTS error: {e}")
        finally:
            # self.publish_face_mode(0)
            # 清理临时文件
            for f in [mp3_path, wav_path]:
                if f and os.path.exists(f):
                    os.remove(f)
            self.get_logger().info("Speech complete.")

    # def publish_face_mode(self, value: int):
    #     msg = Int32()
    #     msg.data = value
    #     self.face_pub.publish(msg)

    def destroy_node(self):
        """清理关闭节点。

        设置停止事件标志，安全地停止异步事件循环，
        等待工作线程结束（超时 1 秒），最后调用父类清理方法。
        """
        self.stop_event.set()                                    # 通知工作线程停止
        self.loop.call_soon_threadsafe(self.loop.stop)           # 线程安全地停止事件循环
        self.worker_thread.join(timeout=1.0)                     # 等待工作线程结束
        super().destroy_node()


def main(args=None):
    """TTS 语音合成节点主函数。

    初始化 ROS2 节点，启动自旋等待消息回调，
    在接收到键盘中断信号时安全关闭。

    Args:
        args: 传递给 rclpy.init() 的命令行参数，默认为 None。
    """
    rclpy.init(args=args)
    node = TTSSpeaker()
    try:
        rclpy.spin(node)  # 进入 ROS2 事件循环，等待消息回调
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()