#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
聊天界面节点 —— 用户与多机器人系统交互的 GUI 入口。
使用 customtkinter 构建桌面聊天窗口，支持文本输入和消息展示。
"""
import json
import os
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Trigger
from ament_index_python.packages import get_package_share_directory

import customtkinter as ctk
from threading import Thread
from datetime import datetime

class ChatGUI(Node):
    """ROS 2 节点：人类聊天界面。发布用户输入到 /chat/input，订阅 /chat/output 显示回复。"""
    def __init__(self):
        super().__init__("human_gui")

        # 发布用户消息到 /chat/input，供 ChatManager 和 TaskManager 处理
        self.publisher = self.create_publisher(String, "/chat/input", 10)
        # 订阅 /chat/output，接收 ChatManager 转发的消息并显示在 GUI 上
        self.subscription = self.create_subscription(String, "/chat/output", self.on_output, 10)
        # 订阅仿真时间，用于在消息旁显示时间戳
        self.time_sub = self.create_subscription(String, "/current_time", self.on_time, 10)

        self.declare_parameter("config_file", "robot_config_assmble_help")
        cfg_file_name = self.get_parameter("config_file").get_parameter_value().string_value
        package_share = get_package_share_directory("chatty")

        cfg_file_name = cfg_file_name + ".json"
        self.config_file_path = os.path.join(package_share, "config", cfg_file_name)

        self.read_json_config()
        self.robot_names = []
        self.robot_colors = []

        # 设置 customtkinter 主题（浅色 + 蓝色）
        ctk.set_appearance_mode("light")
        ctk.set_default_color_theme("blue")

        # 机器人消息颜色调色板 —— 每种机器人类型用不同颜色区分
        self.colors = {
            "bg_light": "#f0f0f0",       # 聊天背景（浅灰）
            "human_msg": "#3F88C5",      # 人类消息（蓝色）
            "task_msg": "#FFA851",       # TaskManager 消息（浅橙色）
            "button_hover": "#ff6f61",   # 按钮悬停色（珊瑚红）
            "go2_msg": '#888888',        # Go2 机器人（灰色）
            "burger_msg": "#008080",     # Burger 机器人（青色）
            "waffle_msg": "#9932a8",     # Waffle 机器人（紫色）
            "drone_msg" : "#4E5734",     # 无人机（暗绿）
            "formation_msg" : "#6B6DE6", # 编队控制（淡紫）
            "x_arm_msg": "#3A9026" ,     # 机械臂（深绿）
            "lerobot1_msg": "#FF69B4",   # LeRobot 1（热粉色）
            "lerobot2_msg": "#80E480",   # LeRobot 2（浅绿色）
            "clock_msg": "#20B2AA",      # 时钟消息（浅海绿）

            # 在此处可为更多机器人或消息类型添加颜色
        }

        # ---- 创建 customtkinter 窗口 ----
        self.window = ctk.CTk()
        self.window.title("CoMuRoS")
        self.window.geometry("700x600")
        self.window.configure(fg_color=self.colors["bg_light"])

        # 顶部标题栏
        self.header_frame = ctk.CTkFrame(self.window, fg_color=self.colors["human_msg"], corner_radius=0)
        self.header_frame.pack(fill='x', pady=(0, 10))

        self.header_label = ctk.CTkLabel(
            self.header_frame,
            text="CHAT INTERFACE",
            font=("Montserrat", 22, "bold"),
            text_color="#ffffff"
        )
        self.header_label.pack(pady=10)

        # 可滚动的聊天消息展示区域
        self.chat_area = ctk.CTkScrollableFrame(self.window, fg_color=self.colors["bg_light"])
        self.chat_area.pack(padx=15, pady=10, fill='both', expand=True)

        # 底部输入区域
        entry_frame = ctk.CTkFrame(self.window, fg_color=self.colors["human_msg"], corner_radius=10, height=70)
        entry_frame.pack(fill='x', padx=15, pady=15)
        entry_frame.pack_propagate(False)

        # 文本输入框
        self.entry = ctk.CTkEntry(
            entry_frame,
            font=("Montserrat", 14, "bold"),
            placeholder_text="在此输入消息...",
            height=35,
            corner_radius=20,
            fg_color="#ffffff",
            text_color="#000000",
        )
        self.entry.pack(side='left', expand=True, fill='x', padx=10, pady=(0, 10))
        self.entry.bind("<Return>", self.send_message)  # 回车发送

        # 发送按钮
        send_btn = ctk.CTkButton(
            entry_frame,
            text="发送",
            command=self.send_message,
            font=("Montserrat", 14, "bold"),
            fg_color="#ff5252",
            hover_color=self.colors["button_hover"],
            corner_radius=20,
            width=100,
            height=35
        )
        send_btn.pack(side='right', padx=10, pady=(0, 10))

        self.get_logger().info("[ChatGUI] GUI node initialized.")

        # 启动后延迟加载历史记录
        self.history_fetched = False
        self.window.after(500, self.fetch_history_once)

    def read_json_config(self):
        """从 JSON 配置文件读取机器人名称和对应的显示颜色。"""
        try:
            # Ensure the file exists before opening
            if not os.path.exists(self.config_file_path):
                self.get_logger().error(f"[ChatGUI] Config file not found: {self.config_file_path}")
                return 

            with open(self.config_file_path, "r") as f:
                config = json.load(f)

            self.get_logger().info(f"[ChatGUI] Loaded config from {self.config_file_path}")

            # --- Extract robot names and colors ---
            robot_names = config.get("robot_names", [])
            colors_assigned = config.get("colors_assigned", {})

            # Create parallel color list (default to white if not found)
            robot_colors = [colors_assigned.get(name, "#FFFFFF") for name in robot_names]

            # Save to instance for later use
            self.robot_names = robot_names
            self.robot_colors = robot_colors

            # Log results for debugging
            self.get_logger().info(f"[ChatGUI] Robots: {self.robot_names}")
            self.get_logger().info(f"[ChatGUI] Colors: {self.robot_colors}")

            # Return lists
            return 

        except Exception as e:
            self.get_logger().error(f"[ChatGUI] Failed to load config: {e}")
            return 

        
    def on_time(self, msg):
        """处理仿真时间回调，更新当前时间。"""
        self.current_time = msg.data

    def on_output(self, msg):
        """
        处理 /chat/output 话题上的消息回调。
        去除时间戳前缀后，在主线程中将消息添加到聊天界面。
        """
        line = msg.data
        self.get_logger().info(f"[ChatGUI] /chat/output -> {line}")

        # Remove `[dd:mm:yy timestamp]` part
        # 去除消息中的时间戳前缀 "[dd:mm:yy]"
        if "]" in line:
            line = line.split("] ", 1)[-1]

        # 使用 after(0) 确保 GUI 更新在主线程中执行
        self.window.after(0, lambda: self.append_text(line))


        
    
    def append_text(self, text_line):
        """
        将一条消息文本渲染到聊天界面。
        根据消息来源（人类/不同机器人/任务管理器）选择对应的颜色和排版方向，
        并添加时间戳标签。
        """
        # Get timestamp in HH:MM format
        timestamp = datetime.now().strftime("%H:%M")
        # if not timestamp :
        #     timestamp = self.current_time

        # 判断消息来源，设置对应的颜色、标签文本和对齐方向
        # "e" 表示右对齐（人类发送），"w" 表示左对齐（机器人/系统回复）
        if "Human:" in text_line:
            message_color = self.colors["human_msg"]
            label_text = "Human"
            text_line = text_line.replace("Human:", "").strip()
            align = "e"

        elif "Burger (msg)" in text_line :
            text_line = text_line.replace("Unknown:", "").strip()
            text_line = text_line.replace("(msg)", "").strip()
            text_line = text_line.replace(":", "").strip()
            message_color = self.colors["burger_msg"]
            label_text = "Burger"
            text_line = text_line.replace("Burger", "").strip()
            align = "w"

        elif "Waffle (msg)" in text_line:
            text_line = text_line.replace("Unknown:", "").strip()
            text_line = text_line.replace("(msg)", "").strip()
            text_line = text_line.replace(":", "").strip()
            message_color = self.colors["waffle_msg"]
            label_text = "Waffle"
            text_line = text_line.replace("Waffle", "").strip()
            align = "w"

        elif "Go2 (msg)" in text_line:
            text_line = text_line.replace("Unknown:", "").strip()
            text_line = text_line.replace(":", "").strip()
            text_line = text_line.replace("(msg)", "").strip()
            message_color = self.colors["go2_msg"]
            label_text = "Go2"
            text_line = text_line.replace("Go2", "").strip()
            align = "w"

        elif "Drone (msg)" in text_line:
            text_line = text_line.replace("Unknown:", "").strip()
            text_line = text_line.replace(":", "").strip()
            text_line = text_line.replace("(msg)", "").strip()
            message_color = self.colors["drone_msg"]
            label_text = "Drone"
            text_line = text_line.replace("Drone", "").strip()
            align = "w"

        elif "Formation (msg)" in text_line:
            text_line = text_line.replace("Unknown:", "").strip()
            text_line = text_line.replace(":", "").strip()
            text_line = text_line.replace("(msg)", "").strip()
            message_color = self.colors["formation_msg"]
            label_text = "Formation"
            text_line = text_line.replace("Formation", "").strip()
            align = "w"

        elif "X Arm (msg)" in text_line:
            text_line = text_line.replace("Unknown:", "").strip()
            text_line = text_line.replace(":", "").strip()
            text_line = text_line.replace("(msg)", "").strip()
            message_color = self.colors["x_arm_msg"]
            label_text = "X Arm"
            text_line = text_line.replace("X Arm", "").strip()
            align = "w"


        elif "Lerobot1 (msg)" in text_line :
            text_line = text_line.replace("Unknown:", "").strip()
            text_line = text_line.replace("(msg)", "").strip()
            text_line = text_line.replace(":", "").strip()
            message_color = self.colors["lerobot1_msg"]
            label_text = "Lerobot1"
            text_line = text_line.replace("Lerobot1", "").strip()
            align = "w"


        elif "Lerobot2 (msg)" in text_line :
            text_line = text_line.replace("Unknown:", "").strip()
            text_line = text_line.replace("(msg)", "").strip()
            text_line = text_line.replace(":", "").strip()
            message_color = self.colors["lerobot2_msg"]
            label_text = "Lerobot2"
            text_line = text_line.replace("Lerobot2", "").strip()
            align = "w"


        elif "Clock (msg)" in text_line :
            text_line = text_line.replace("Unknown:", "").strip()
            text_line = text_line.replace("(msg)", "").strip()
            text_line = text_line.replace(":", "").strip()
            message_color = self.colors["clock_msg"]
            label_text = "Clock"
            text_line = text_line.replace("Clock", "").strip()
            align = "w"


        elif "Task Manager" in text_line or "Unknown" in text_line :
            text_line = text_line.replace("Task Manager:", "").strip()
            message_color = self.colors["task_msg"]
            label_text = "Task Manager:"
            text_line = text_line.replace("Taskmanager:", "").strip()
            align = "w"

        else:
            # 未匹配到已知消息来源，尝试在机器人配置列表中查找
            self.get_logger().error(f"Checking for robot name: (msg): in {text_line}")

            for name in self.robot_names:
                # normailizing is need to be updated
                self.get_logger().error(f"Checking for robot name: {name.capitalize() } (msg): in {text_line}")
                if f"{name.capitalize()} (msg)" in text_line:
                    self.get_logger().info("\n\n\n\n\n\n\n Match found \n\n\n\n\n\n\n")
                    text_line = text_line.replace("Unknown:", "").replace("(msg)", "").replace(":", "").strip()
                    message_color = self.colors.get(f"{name.lower()}_msg", "#000000")
                    label_text = name.capitalize()
                    text_line = text_line.replace(name, "").strip()
                    align = "w"
                    break

            else:
                # 仍未匹配，默认显示为系统消息
                self.get_logger().error("\n\n\n\n\n\n\n No match, defaulting to System\n\n\n\n\n\n\n")
                message_color = "#ffffff"
                label_text = "System"
                align = "w"


        # Message frame — 消息气泡框架
        message_frame = ctk.CTkFrame(self.chat_area, fg_color=message_color, corner_radius=10)
        message_frame.pack(fill="x", padx=10, pady=5, anchor=align)

        # Label (Sender + Timestamp) — 发送者标签 + 时间戳
        label = ctk.CTkLabel(
            message_frame,
            text=f"{label_text} • {timestamp}",
            font=("Montserrat", 15, "bold"),
            text_color="#ffffff",
            justify="left"
        )
        label.pack(anchor="w", padx=10, pady=(5, 0))

        # Message text — 消息正文
        message_label = ctk.CTkLabel(
            message_frame,
            text=text_line,
            font=("Montserrat", 14, "bold"),
            text_color="#ffffff",
            wraplength=500,
            justify="left"
        )
        message_label.pack(padx=10, pady=(0, 5), anchor="w")

        # 刷新聊天区域并自动滚动到底部
        self.chat_area.update_idletasks()
        self.chat_area._parent_canvas.yview_moveto(1)

    def send_message(self, event=None):
        """
        发送用户输入的消息。
        从输入框获取文本，包装为 "human|..." 格式的消息，
        发布到 /chat/input 话题供 ChatManager 处理。
        """
        user_input = self.entry.get().strip()
        if user_input:
            out_msg = String()
            out_msg.data = f"human|{user_input}"
            # out_msg.data = f"Human (msg) | {user_input}"
            self.publisher.publish(out_msg)
            self.get_logger().info(f"[ChatGUI] Sent -> {user_input}")
        # 发送后清空输入框
        self.entry.delete(0, 'end')

    def fetch_history_once(self):
        """
        在 GUI 启动后一次性获取历史聊天记录。
        通过 ROS 2 服务 "get_chat_history" 向 ChatManager 请求历史消息，
        并将返回的消息依次渲染到聊天区域。
        """
        if self.history_fetched:
            return
        self.get_logger().info("[ChatGUI] Attempting to fetch old chat.")
        # 创建 ROS 2 服务客户端
        client = self.create_client(Trigger, "get_chat_history")
        if not client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn("[ChatGUI] Manager not ready, skipping old history fetch.")
            self.history_fetched = True
            return

        req = Trigger.Request()
        future = client.call_async(req)

        def check_done():
            """轮询检查异步服务调用是否完成。"""
            if future.done():
                res = future.result()
                if res and res.success:
                    lines = res.message.split("\n")
                    self.get_logger().info(f"[ChatGUI] Received {len(lines)} lines of old chat.")
                    for line in lines:
                        if line.strip():
                            # 将每行历史消息添加到聊天界面
                            self.window.after(0, lambda l=line: self.append_text(l))
                else:
                    self.get_logger().error("[ChatGUI] Could not fetch old history or manager gave error.")
                self.history_fetched = True
            else:
                # 未完成则 200ms 后再次检查
                self.window.after(200, check_done)

        self.window.after(200, check_done)

    def run_gui(self):
        """启动 customtkinter 主事件循环，显示聊天窗口。"""
        self.get_logger().info("[ChatGUI] Starting Tkinter mainloop.")
        self.window.mainloop()


def main():
    """
    入口函数。
    初始化 ROS 2，创建 ChatGUI 节点，在后台线程中执行 rclpy.spin 以处理 ROS 回调，
    在主线程中运行 GUI 事件循环。
    """
    rclpy.init()
    node = ChatGUI()

    def spin_bg():
        """后台线程函数：持续处理 ROS 2 回调。"""
        rclpy.spin(node)

    t = Thread(target=spin_bg, daemon=True)
    t.start()

    try:
        node.run_gui()
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()