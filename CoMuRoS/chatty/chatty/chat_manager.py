#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
聊天管理器节点 —— 负责消息路由、聊天记录存储和检索。
订阅 /chat/input 接收消息，处理后发布到 /chat/output，
并提供历史记录查询服务。
"""
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Trigger
from ament_index_python.packages import get_package_share_directory
import os

class ChatManager(Node):
    """
    ChatManager Node:
      - Routes messages between users and robots.
      - Stores chat history and provides it on request.
      - Loads previous history on startup and appends new messages on shutdown.

    消息路由节点：在用户和机器人之间转发消息，
    存储聊天历史并提供查询服务，
    启动时加载旧历史，关闭时追加新消息。
    """

    def __init__(self):
        """初始化节点：设置订阅/发布、服务、文件路径，清理当前会话历史。"""
        super().__init__("chat_manager")
        # 聊天记录列表，按时间顺序存储每条消息
        self.chat_log = []
        self.current_time = f"Hours: {00}, Minutes: {00}, Seconds: {00}"

        self.get_logger().info("[ChatManager] Initializing...")

        # Load previous chat history from file
        # 从文件加载历史记录（暂时注释掉）
        # self.load_history()

        # Subscriptions and Publishers
        # 订阅 /chat/input 接收用户/系统的输入消息
        self.input_sub = self.create_subscription(String, "/chat/input", self.handle_input, 10)
        # 发布格式化后的消息到 /chat/output，供其他节点（如 ChatGUI）消费
        self.output_pub = self.create_publisher(String, "/chat/output", 10)
        # 发布完整聊天历史到 /chat/history
        self.history_pub = self.create_publisher(String, "/chat/history", 10)
        # 订阅仿真时间话题
        self.timesub = self.create_subscription(String, "/current_time", self.handle_time, 10)


        # History Retrieval Service — 提供历史记录查询服务
        self.history_service = self.create_service(Trigger, "get_chat_history", self.handle_history)

        package_name = "chatty"
        directry = "data"
        package_path = get_package_share_directory(package_name)

        # 持久化历史文件路径（跨实验）
        script_name = "chat_history.txt"
        self.file_path = os.path.join(package_path, directry, script_name)

        # 当前实验历史文件路径
        script_name_current = "chat_history_current.txt"
        self.file_path_current = os.path.join(package_path, directry, script_name_current)

        # 启动时清空当前实验的临时历史文件
        self.clean_current_history()

        self.get_logger().info("[ChatManager] Ready and running.")

    def load_history(self):
        """Loads previous chat history from file.
        从持久化历史文件中加载之前的聊天记录。"""
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, "r") as file:
                    self.chat_log = file.read().splitlines()
                self.get_logger().info(f"[ChatManager] Loaded {len(self.chat_log)} previous messages.")
            except Exception as e:
                self.get_logger().error(f"[ChatManager] Failed to load history: {e}")

    def handle_time(self, msg):
        """Handles incoming time messages.
        处理 /current_time 话题的时间消息，更新当前时间。"""
        # Extract time from message
        self.current_time = msg.data
        # self.get_logger().info(f"[ChatManager] Current time updated to {self.current_time}")


    def handle_input(self, msg:String):
        """Handles incoming chat messages and distributes them.
        处理 /chat/input 上的消息：
        解析 "role|content" 格式，添加时间戳，存入历史记录，
        发布到 /chat/output 和 /chat/history，并保存到当前历史文件。"""
        # timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # if not timestamp :
            # timestamp = self.current_time

        timestamp = self.current_time
        # 以 "|" 分隔角色和内容
        parts = msg.data.split("|", 1)

        if len(parts) == 2:
            # 格式: "human|消息内容" 或 "robot_name|消息内容"
            role, content = parts[0].strip(), parts[1].strip()
            self.chat_entry = f"[Time: {timestamp}] {role.capitalize()}: {content}"
        else:
            # 没有 "|" 分隔符，默认作为 Task Manager 消息
            role, content = "Task Manager", msg.data
            # role, content = "Task Manager (msg)", msg.data
            self.chat_entry = f"[Time: {timestamp}] {role}:\n{content}"

        # self.chat_entry = f"[Time: {timestamp}] {role.capitalize()}: {content}"


        self.get_logger().info(f"[ChatManager] Received on /input-> {self.chat_entry}")

        # Store in chat history — 存入历史记录列表
        self.chat_log.append(self.chat_entry)

        # Publish to output topic — 发布格式化消息到 /chat/output
        out_msg = String()
        out_msg.data = self.chat_entry
        self.output_pub.publish(out_msg)

        # Publish full history — 发布完整聊天历史到 /chat/history
        history_msg = String()
        history_msg.data = "\n".join(self.chat_log)
        self.history_pub.publish(history_msg)
        # 追加到当前实验的历史文件
        self.save_history_current()

    def handle_history(self, request, response):
        """Handles chat history requests from clients.
        处理 "get_chat_history" 服务请求，返回全部聊天历史。"""
        response.success = True
        response.message = "\n".join(self.chat_log)
        self.get_logger().info(f"[ChatManager] Returning {len(self.chat_log)} chat entries.")
        return response

    def save_history(self):
        """Appends new chat messages to history file on shutdown.
        在节点关闭时将本次实验的聊天记录追加到持久化历史文件中。"""
        try:
            with open(self.file_path, "a") as file:
                file.write(f"****************   NEW EXPERIMENT   ********************** \n")
                for line in self.chat_log:
                    file.write(line + "\n")
            self.get_logger().info(f"[ChatManager] Appended {len(self.chat_log)} new messages to {self.file_path}.")
        except Exception as e:
            self.get_logger().error(f"[ChatManager] Failed to save history: {e}")

    def clean_current_history(self):
        """Cleans the current chat history file on startup.
        在节点启动时清空当前实验的临时历史文件，准备记录新的会话。"""
        try:
            with open(self.file_path_current, "w") as file:
                file.write("*********     NEW EXPERIMNET           ********* \n")  # Clear the file 清空文件并写入新实验标记
            self.get_logger().info(f"[ChatManager] Cleared current history file at {self.file_path_current}.")

        except Exception as e:
            self.get_logger().error(f"[ChatManager] Failed to clear current history: {e}")

    def save_history_current(self):
        """Appends new chat messages to current history file on chat output.
        每当有新的聊天消息产生时，将其追加到当前实验的临时历史文件中。"""
        try:
            with open(self.file_path_current, "a") as file:
                file.write(self.chat_entry + "\n")
            self.get_logger().info(f"[ChatManager] Appended {len(self.chat_log)} new messages to {self.file_path_current}.")

        except Exception as e:
            self.get_logger().error(f"[ChatManager] Failed to save history: {e}")

    def destroy_node(self):
        """Handles cleanup before shutdown.
        重写父类的 destroy_node 方法，在节点销毁前保存聊天历史。"""
        self.save_history()
        super().destroy_node()


def main():
    """Entry point for the ChatManager node.
    入口函数：初始化 ROS 2，创建 ChatManager 节点，保持运行直到收到中断信号。"""
    rclpy.init()
    node = ChatManager()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("[ChatManager] Shutting down...")
        node.get_logger().info("Keyboard interrupt received. Shutting down.")
    finally:
        # 清理节点并保存历史
        node.destroy_node()
        # node.save_history()
        rclpy.shutdown()

if __name__ == "__main__":
    main()