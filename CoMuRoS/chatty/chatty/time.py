#!/usr/bin/env python3
"""
仿真时间发布节点 —— 定期发布系统运行的模拟时间信息。

计算从节点启动以来的运行时长，并以 HH:MM:SS 格式发布到
/current_time 话题，为多机器人协同系统中的其他节点提供统一的
时间参考。
"""
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import time

class TimePublisher(Node):
    """时间发布节点。

    使用系统时钟（time.time()）计算从节点启动到当前的运行时长，
    通过定时器周期性发布格式化的时间字符串到 /current_time 话题。
    可用于仿真系统中的时间同步和时间显示。
    """
    def __init__(self):
        """初始化时间发布节点。

        创建 /current_time 话题的发布器，设置定时器周期为 1 秒，
        记录启动时间戳用于计算运行时长。
        """
        super().__init__("time_publisher")
        self.publisher_ = self.create_publisher(String, "/current_time", 10)
        self.timer_period = 1.0  # 发布周期（秒）
        self.start_time = time.time()  # 记录节点启动时间
        self.timer = self.create_timer(self.timer_period, self.timer_callback)
        self.get_logger().info("TimePublisher node has been started.")

    def timer_callback(self):
        """定时器回调函数：计算并发布运行时间。

        计算从节点启动到当前时刻的累计运行时长，
        将其格式化为 "Hours: XX, Minutes: XX, Seconds: XX" 格式，
        并发布到 /current_time 话题。
        """
        msg = String()
        elapsed = int(time.time() - self.start_time)
        hours, rem = divmod(elapsed, 3600)
        minutes, seconds = divmod(rem, 60)
        # 将运行时长格式化为 HH:MM:SS 风格字符串
        msg.data = f"Hours: {hours:02d}, Minutes: {minutes:02d}, Seconds: {seconds:02d}"
        self.publisher_.publish(msg)
        # self.get_logger().info(f"Published: '{msg.data}'")

def main(args=None):
    """时间发布节点主函数。

    初始化 ROS2 节点，创建 TimePublisher 并进入自旋循环，
    周期性发布仿真运行时间。处理键盘中断实现优雅关闭。

    Args:
        args: 传递给 rclpy.init() 的命令行参数，默认为 None。
    """
    rclpy.init(args=args)
    time_publisher = TimePublisher()
    try:
        rclpy.spin(time_publisher)
    except KeyboardInterrupt:
        time_publisher.get_logger().info("Keyboard interrupt detected. Shutting down.")
    finally:
        time_publisher.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()