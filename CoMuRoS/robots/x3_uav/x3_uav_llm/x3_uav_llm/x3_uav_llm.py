#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
无人机LLM控制节点模块（Drone LLM Node Module）

该模块实现了基于大语言模型（LLM）的四旋翼无人机控制节点。
RobotLLMNode类作为一个中枢节点，负责：
  - 监听聊天主题，接收任务指令
  - 调用OpenAI GPT-4o/VLM生成控制代码或视觉分析
  - 执行生成的Python代码控制无人机飞行（悬停、移动到指定位置）
  - 使用底部摄像头进行视觉场景描述
  - 发布任务状态更新

主要功能：
  - 无人机goto_service：移动到指定3D位置和偏航角
  - 视觉查询：使用GPT-4.1视觉模型分析底部摄像头图像
  - 任务取消机制
"""

import json
import time

import openai
import base64
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from sensor_msgs.msg import CompressedImage
from dataclasses import dataclass
import os
from rclpy.callback_groups import (
    MutuallyExclusiveCallbackGroup,
    ReentrantCallbackGroup
)
from ament_index_python.packages import get_package_share_directory

from robot_interface.srv import GotoPoseDrone, Find
from openai import OpenAI


ROBOT_NAME   = 'drone'
ROBOT_TYPE   = "Quadrotor UAV"
NODE_NAME    = "drone_llm_node"
PACKAGE_NAME = "x3_uav_llm"

api_key = os.getenv("OPENAI_API_KEY")


# 定义无人机可用的动作选项
@dataclass
class TestOption:
    name: str           # 动作名称
    id: int             # 动作唯一标识
    description: str    # 动作描述，供LLM选择参考
    example_code: str   # 示例调用代码，LLM将参考此格式生成代码

option_list = [
    TestOption(
        name="Hover",
        id=0,
        description="This is used to make the drone hover or it can be used to goto the specified position",
        example_code="node.hover(x=1.0, y=2.0, z=3.0, yaw_deg=0.0)"
    ),
    TestOption(
        name="DescribeScreen",
        id=1,
        description="This is used to describe the current screen view of the drone given a prompt. Answer the prompt in only one go. ",
        example_code="node.describe_screen(prompt='in which table is child is sitting?')"
    ),
]

class RobotLLMNode(Node):
    """
    无人机LLM控制节点

    功能概述：
      - 监听聊天话题（/chat/output等），接收用户任务指令
      - 发布任务状态更新（/chat/task_status 和 <robot_name>_task_status）
      - 接收/处理机器人状态消息（/robot_states，JSON格式）
      - 调用OpenAI API生成Python飞行控制代码
      - 使用GPT-4.1视觉模型分析底部摄像头图像
      - 支持任务取消机制
    """

    _instance = None  # 单例实例引用

    def __init__(self) -> None:
        """初始化无人机LLM控制节点"""
        super().__init__(NODE_NAME)

        # ---- 参数声明 ----
        self.declare_parameter('robot_name', ROBOT_NAME)
        self.robot_name: str = self.get_parameter('robot_name').value
        robot_task_topic = f'{self.robot_name}_task_status'

        RobotLLMNode._instance = self

        self.current_time = f"Hours: {00}, Minutes: {10}, Seconds: {00}"
        self.robot_task = ""           # 当前任务描述
        self.robot_states = {}         # 机器人状态字典

        # OpenAI视觉客户端（用于VLM图像分析）
        self.Visionclient = OpenAI(api_key=api_key)
        self.latest_image_b64 = None   # 最新摄像头图像的Base64编码

        # 回调组配置
        self.single_group = MutuallyExclusiveCallbackGroup()  # 互斥回调组（聊天等独占操作）
        self.seq_group = MutuallyExclusiveCallbackGroup()     # 顺序执行回调组
        self.multi_group = ReentrantCallbackGroup()           # 可重入回调组（支持并发）

        # ---- 发布者 ----
        self.pub_task_status = self.create_publisher(String, '/chat/task_status', 10)
        self.pub_input_msg = self.create_publisher(String, '/chat/input', 10)
        self.pub_robot_states = self.create_publisher(String, '/robot_states', 10)
        self.pub_robot_task = self.create_publisher(String, robot_task_topic, 10)

        # 无人机位置控制服务客户端
        self._goto_client = self.create_client(GotoPoseDrone, '/r3/goto_pose', callback_group=self.multi_group)
        # 取消goto指令的发布者
        self._cancel_goto_pub = self.create_publisher(Bool, '/r3/cancel_goto_pose_goal', 10)

        # ---- 订阅者 ----
        self.sub_robot_states = self.create_subscription(
            String, '/robot_states', self.on_robot_states, 10,
            callback_group=self.seq_group
        )
        self.current_time_sub = self.create_subscription(
            String, '/current_time', self.on_current_time, 10,
            callback_group=self.seq_group
        )
        self.sub_tasks_json = self.create_subscription(
            String, '/task_manager/tasks_json', self.on_tasks_json, 10,
            callback_group=self.multi_group
        )
        self.sub_chat_output = self.create_subscription(
            String, '/chat/output', self.on_chat_output, 10,
            callback_group=self.single_group
        )

        self.sub_chat_task_status = self.create_subscription(
            String, '/chat/task_status', self.on_chat_task_status, 10
        )

        # 订阅无人机底部摄像头压缩图像
        self.create_subscription(
            CompressedImage,
            "/r3/bottom_camera/color/image_raw/compressed",
            self.image_callback, 10,
            callback_group=self.single_group
        )

        self.get_logger().info(
            f'RobotLLMNode started for robot="{self.robot_name}". '
            f'Publishing robot task status on "{robot_task_topic}".'
        )

        # 设置历史记录文件路径
        directry = "data"
        package_path = get_package_share_directory(PACKAGE_NAME)

        script_name = self.robot_name+"_chat_history.txt"
        self.history_file = os.path.join(package_path, directry, script_name)

        script_name = self.robot_name+'_task_history.txt'
        self.robot_task_history = os.path.join(package_path, directry, script_name)

        self.clear_files()
        self.robot_has_no_current_task()


    @classmethod
    def get_instance(cls):
        """安全地获取单例节点实例"""
        if cls._instance is None:
            raise RuntimeError("RobotLLMNode has not been created yet! Did you run the node?")
        return cls._instance

    def clear_files(self) -> None:
        """启动时清空对话历史文件和任务历史文件"""
        if os.path.exists(self.history_file):
            with open(self.history_file, "w") as file:
                file.write("")
            self.get_logger().info("Cleared chat history file on startup.")
        else:
            with open(self.history_file, "w") as file:
                file.write("")
            self.get_logger().warn(f"Chat history file not found. Created new file: {self.history_file}")

        if os.path.exists(self.robot_task_history):
            with open(self.robot_task_history, "w") as file:
                file.write("")
            self.get_logger().info("Cleared the robot task history file on startup.")
        else:
            self.get_logger().warn(f"Robot Task history file not found: {self.robot_task_history}")
            with open(self.robot_task_history, "w") as file:
                file.write("")
            self.get_logger().warn(f"Robot task history file not found. Created new file: {self.robot_task_history}")

    # -------------------- 回调函数 --------------------

    def image_callback(self, msg: CompressedImage):
        """接收底部摄像头压缩图像并转换为Base64编码存储"""
        try:
            self.latest_image_b64 = base64.b64encode(msg.data).decode("utf-8")
        except Exception as e:
            self.get_logger().error(f"Error converting image: {e}")

    def on_robot_states(self, msg: String) -> None:
        """处理机器人状态更新（msg.data预期为JSON字符串）"""
        try:
            data = json.loads(msg.data)
            rs = data.get("robot_states")
            if isinstance(rs, dict):
                self.robot_states = rs
                self.get_logger().debug(f"robot_states keys: {list(self.robot_states.keys())}")
            else:
                self.get_logger().warning("/robot_states missing or not a dict; ignoring payload.")
        except json.JSONDecodeError:
            self.get_logger().error(f"/robot_states not JSON: {msg.data}")

    def on_tasks_json(self, msg: String) -> None:
        """处理来自任务管理器的任务JSON，解析并执行分配给无人机的任务"""
        self.get_logger().debug(f'Received /task_manager/tasks_json: {msg.data}')
        try:
            data = json.loads(msg.data)
            robot_tasks = {
                key.lower().replace(" ", "_"): value
                for key, value in data.get("robot_tasks", {}).items()
            }

            robot_name = self.robot_name

            robot_names = list(robot_tasks.keys())
            for name in robot_names:
                if self.robot_name in name:
                    robot_name = name
                    break

            robot_task = robot_tasks.get(robot_name, "").strip()
            self.get_logger().debug(f"{self.robot_name} task fetched [1]")

            if not robot_task:
                robot_task = robot_tasks.get(self.robot_name+'_task', "").strip()
                self.get_logger().debug(f"{self.robot_name+'_task'} task fetched [2]")

            if not robot_task:
                robot_task = robot_tasks.get(self.robot_name+'_tasks', "").strip()
                self.get_logger().debug(f"{self.robot_name+'_tasks'} task fetched [3]")

            if not robot_task:
                robot_task = f"No {self.robot_name} task found."
                self.get_logger().debug(robot_task)
                return
            
            elif "stop" in robot_task.lower():
                self.get_logger().info(f"{self.robot_name} task: {robot_task}")
                self.robot_task_interrupted(robot_task)
                self.stop_tasks()
                msg.data = f'{self.robot_name.capitalize()} (status): STOP TASKS COMPLETED'
                self.pub_task_status.publish(msg)
                return

            self.robot_task = robot_task
            
            self.get_logger().info(f"{self.robot_name} task: {robot_task}")
            
            self.get_logger().info("Robot Task in Progress ..")
            self.robot_task_in_progress(robot_task)

            self.get_logger().info("Excuting Robot Task ..")
            self.execute_task(robot_task)
            self.get_logger().info(f'On Task Json Task executed')

            # status.data = f'{self.robot_name}: received {len(tasks) if isinstance(tasks, list) else 1} task set(s)'
        except json.JSONDecodeError:
            self.get_logger().warn(
                f'Received /task_manager/tasks_json with invalid JSON; raw: {msg.data}'
            )
            # status.data = f'{self.robot_name}: received invalid tasks JSON'
        # self.pub_task_status.publish(status)

    def on_chat_output(self, msg: String) -> None:
        """处理原始聊天输出内容，并保存到历史记录文件中"""
        self.get_logger().debug(f'Received /chat/output: {msg.data}')

        timestamp = self.current_time
        parts = msg.data.split("|", 1)

        # 解析消息格式："角色|内容"，如果格式正确则格式化存储
        if len(parts) == 2:
            role, content = parts[0].strip(), parts[1].strip()
            self.chat_entry = f"[Time: {timestamp}] {role.capitalize()}: {content}"
        else:
            role, content = "Task Manager", msg.data
            self.chat_entry = f"[Time: {timestamp}] {role}:\n{content}"

        with open(self.history_file, "a") as file:
            file.write(self.chat_entry + "\n")

    def on_chat_task_status(self, msg: String) -> None:
        """监听外部任务状态变化并记录到历史文件"""
        self.get_logger().debug(f'Observed /chat/task_status: {msg.data}')
        with open(self.history_file, "a") as file:
            file.write(msg.data + "\n")

    def on_current_time(self, msg: String) -> None:
        """从/current_time话题更新时间"""
        self.get_logger().debug(f'Received /current_time: {msg.data}')
        self.current_time = msg.data

    # -------------------- 辅助方法 --------------------

    def update_robot_state(self, incoming: dict, robot_name: str | None = None) -> None:
        """
        将部分机器人状态更新合并到 self.robot_states 中

        - self.robot_states 可能为：
            * None（未初始化）
            * 没有 "robot_states" 键的字典（仅包含元数据如日期）
            * 已经包含 "robot_states" 键的字典
            * 来自旧代码的裸机器人状态字典（会被自动包装）

        - 确保顶层 "robot_states" 键存在
        - 确保每个机器人的子字典存在
        - 添加新键并更新已变化的键值
        """

        # 如果未指定机器人名称，默认为当前节点对应的机器人
        if robot_name is None:
            robot_name = self.robot_name

        # 1) 确保顶层容器存在
        if self.robot_states is None:
            self.robot_states = {}
            self.get_logger().info("Created top-level robot_states container dict")

        # 2) 确保顶层有 "robot_states" 键
        if "robot_states" not in self.robot_states:
            # 启发式判断：如果当前字典看起来只有机器人条目，则将其包装到 "robot_states" 下
            if all(isinstance(v, dict) for v in self.robot_states.values()) and len(self.robot_states) > 0:
                self.robot_states = {"robot_states": self.robot_states}
                self.get_logger().debug("Wrapped existing dict under 'robot_states'")
            else:
                # 创建新的 robot_states 字典，同时保留其他元数据字段
                self.robot_states["robot_states"] = {}
                self.get_logger().info("Created 'robot_states' key in top-level dict")

        robots_dict = self.robot_states["robot_states"]

        # 3) 确保机器人条目存在
        if robot_name not in robots_dict or not isinstance(robots_dict[robot_name], dict):
            robots_dict[robot_name] = {}
            self.get_logger().info(f"Created new robot entry '{robot_name}'")

        saved_state = robots_dict[robot_name]

        # 4) 将传入的键值合并到该机器人的状态中
        for key, new_val in incoming.items():
            if key not in saved_state:
                saved_state[key] = new_val
                self.get_logger().debug(f"{robot_name}: Added '{key}' = '{new_val}'")
            else:
                old_val = saved_state[key]
                if old_val != new_val:
                    saved_state[key] = new_val
                    self.get_logger().debug(f"{robot_name}: Updated '{key}' from '{old_val}' → '{new_val}'")

        # 5) 发布更新后的机器人状态
        msg = String()
        msg.data = json.dumps(self.robot_states)
        self.pub_robot_states.publish(msg)

        # 6) 记录更新后的顶层结构
        self.get_logger().debug(f"robot_states now: {self.robot_states['robot_states']}")

        return

    def robot_task_in_progress(self, robot_task) -> None:
        """发布任务进行中的状态"""
        if not robot_task:
            robot_task = self.robot_task

        task_in_progress_msg = f"{self.robot_name.capitalize()} (status) : {robot_task} : TASK IN PROGRESS"
        self.get_logger().info(task_in_progress_msg)
        self.robot_task_status_update(task_in_progress_msg)
        return

    def robot_task_completed(self, robot_task) -> None:
        """发布任务已完成的状态"""
        if not robot_task:
            robot_task = self.robot_task

        task_completed_msg = f"{self.robot_name.capitalize()} (status) : {robot_task} : TASK COMPLETED"
        self.get_logger().info(task_completed_msg)
        self.robot_task_status_update(task_completed_msg)
        return

    def robot_task_interrupted(self, robot_task) -> None:
        """发布任务被中断的状态"""
        if not robot_task:
            robot_task = self.robot_task

        task_interrupted_msg = f"{self.robot_name.capitalize()} (status) : {robot_task} : TASK INTERRUPTED"
        self.get_logger().info(task_interrupted_msg)
        self.robot_task_status_update(task_interrupted_msg)
        return

    def robot_has_no_current_task(self) -> None:
        """发布当前无任务的状态"""
        robot_task = ""
        no_task_msg = f"{self.robot_name.capitalize()} (status) : {robot_task} : NO CURRENT TASK"
        self.get_logger().info(no_task_msg)
        self.robot_task_status_update(no_task_msg)
        return

    def robot_task_status_update(self, status_msg: str) -> None:
        """更新任务状态：发布到话题并追加到历史文件"""
        msg = String()
        msg.data = status_msg
        self.pub_robot_task.publish(msg)

        try:
            with open(self.robot_task_history, "a") as file:
                file.write(f"{status_msg}\n")
        except FileNotFoundError as e:
            self.get_logger().warn(f"Robot Task history file not found: {e}")

        self.get_logger().info("Robot task status updated... ")
        return

    def tasks_completed(self, task) -> None:
        """发布所有任务已完成的状态"""
        msg = String()
        msg.data = f"{self.robot_name.capitalize()} (status): ALL TASKS COMPLETED"
        self.pub_task_status.publish(msg)
        return

    def stop_tasks(self) -> None:
        """停止所有机器人任务：发送取消信号"""
        self.get_logger().info("Stopping all robot tasks...")

        msg = Bool()
        msg.data = True
        self._cancel_goto_pub.publish(msg)

        self.get_logger().info("Cancel request sent to /goto/cancel")
        self.get_logger().info("All tasks of the robot have been stopped.")

    def read_chat_history(self) -> str:
        """
        从持久化文件中读取完整的对话历史

        返回:
            str: 完整的对话历史字符串，或文件缺失/为空时的提示信息
        """
        self.get_logger().debug(f"Attempting to read chat history from: {self.history_file}")

        if not os.path.exists(self.history_file):
            self.get_logger().debug(f"Chat history file not found: {self.history_file}")
            return "No previous chat history."

        if not os.path.isfile(self.history_file):
            self.get_logger().warn(f"Chat history path exists but is not a file: {self.history_file}")
            return "No previous chat history."

        try:
            with open(self.history_file, "r", encoding="utf-8") as file:
                history = file.read().strip()

            if not history:
                self.get_logger().debug("Chat history file is empty.")
                return "No previous chat history."

            self.get_logger().info("Successfully loaded chat history.")
            return history

        except PermissionError:
            self.get_logger().error(f"Permission denied when reading chat history file: {self.history_file}")
            return "Error: Unable to read chat history (permission denied)."
        except OSError as e:
            self.get_logger().error(f"OS error while reading chat history file: {e}")
            return "Error: Failed to read chat history due to system issue."
        except Exception as e:
            self.get_logger().error(f"Unexpected error reading chat history: {type(e).__name__}: {e}")
            return "Error: Failed to load chat history."

    def generate_action_prompt(self, prompt: str, task: str) -> str:
        """
        调用OpenAI GPT-4o生成无人机控制代码

        参数:
            prompt: 系统提示词，描述无人机能力和可用操作
            task: 用户任务描述

        返回:
            (code, explanation): 生成的Python代码和解释说明
        """
        self.get_logger().info(f"Generating code...")

        try:
            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {'role': 'system', 'content': prompt},
                    {'role': 'user', 'content': task}
                ],
                max_tokens=500,
                temperature=0.5,
            )
            self.get_logger().debug(f"LLM Response: {response}")

            raw = response.choices[0].message.content.strip()
            self.get_logger().info(f"Content: {raw}")
            # 解析LLM返回的代码块（```python ... ``` 格式）
            if "```python" in raw:
                parts = raw.split("```python")
                explaination = parts[0].strip()
                code = parts[1].split("```")[0].strip()
                return code, explaination

            return "", ""

        except openai.Timeout:
            self.get_logger().error("OpenAI request TIMED OUT (no response in time)")
        except openai.AuthenticationError:
            self.get_logger().error("OpenAI authentication failed — check your API key!")
        except openai.RateLimitError:
            self.get_logger().error("OpenAI rate limit hit — slow down or upgrade plan")
        except openai.APIError as e:
            self.get_logger().error("OpenAI API error: %s", e)
        except Exception as e:
            self.get_logger().error("Unexpected error in LLM call: %s: %s", type(e).__name__, e)

        self.get_logger().warn("Returning (None, None) due to LLM failure")
        return None, None

    def execute_task(self, task: str) -> None:
        """
        执行给定的无人机任务：通过LLM生成控制代码并执行

        流程：
        1. 读取对话历史作为上下文
        2. 构建系统提示词，包含可用操作和机器人状态
        3. 调用OpenAI生成控制代码
        4. 通过exec执行生成的Python代码
        """
        chat_history = self.read_chat_history()

        try:
            self.get_logger().info("Building system action messages")
            # 构建可用操作列表供LLM参考
            available_actions = "\n".join(
                [f"Function Name: {opt.name} \nFunction Description: {opt.description} (e.g., {opt.example_code})" for opt in option_list]
            )

            self.get_logger().debug(f"Action message: {available_actions}")
            self.get_logger().info("Building system messages")
            # 构建LLM提示词，包含无人机信息、对话历史、状态和可用操作
            prompt = (
                f"You are a robot control system controlling a {ROBOT_TYPE} named '{self.robot_name}'. "
                "You can generate python code to perform actions. "
                "Based on the given task, you need to choose the appropriate action from the available options. "
                f"Recent Tasks (History): {chat_history} "
                f"Current States of All Robots: {self.robot_states} "
                f"Available Actions: {available_actions} "
                "Using the class reference name same as the example is important. "
                "Use the name 'node' to refer to the RobotLLMNode instance. "
            )

            self.get_logger().debug(f"Prompt message: {prompt}")

        except Exception as e:
            self.get_logger().warning(f"{e}")

        code, explanation = self.generate_action_prompt(prompt, task)

        if code:
            self.get_logger().info(f"Generated Code:\n{code}")
            if explanation:
                self.get_logger().info(f"Explanation:\n{explanation}")

            self.get_logger().info("Calling execute_python_code...")
            execute_python_code(code, node=self)

            self.get_logger().info("Robot Task Completed.")
            self.robot_task_completed(task)
            self.tasks_completed(task)
        else:
            self.get_logger().error("Failed to generate valid code for the task.")
            self.robot_task_interrupted(task)

    def goto_service(self, x: float, y: float, z: float, yaw_deg: float) -> bool:
        """
        调用无人机GoTo位置控制服务（非阻塞，无嵌套spin）

        参数:
            x, y, z: 目标位置坐标（米）
            yaw_deg: 目标偏航角（度）

        返回:
            bool: 移动是否成功
        """
        self.get_logger().info(f"Sending drone goto goal: x={x}, y={y}, z={z}, yaw={yaw_deg}°")

        if not self._goto_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("GotoPoseDrone service NOT available!")
            return False

        req = GotoPoseDrone.Request()
        req.x = x
        req.y = y
        req.z = z
        req.yaw_deg = yaw_deg

        future = self._goto_client.call_async(req)
        self.get_logger().info("Waiting for drone service response...")

        deadline = time.time() + 1000000.0
        while rclpy.ok() and not future.done():
            time.sleep(0.05)
            if time.time() > deadline:
                self.get_logger().error("Drone goto service call TIMED OUT.")
                return False

        try:
            res = future.result()
        except Exception as e:
            self.get_logger().error(f"Drone goto service call failed: {e}")
            return False

        if not res.accepted:
            self.get_logger().warn(f"Drone goto request was NOT accepted: {res.message}")
            return False

        if res.success:
            self.get_logger().info(f"Drone goto SUCCESS: {res.message}")
            return True
        else:
            self.get_logger().warn(f"Drone goto FAILED: {res.message}")
            return False

    def query_callback(self, prompt: str) -> bool:
        """
        使用GPT-4.1视觉模型分析底部摄像头图像

        参数:
            prompt: 视觉查询提示词

        返回:
            bool: 查询是否成功
        """
        if self.latest_image_b64 is None:
            self.get_logger().warn("No image received yet. Cannot query VLM.")
            return False

        self.get_logger().info("Sending image to VLM...")

        # 系统提示词：定义视觉分析的角色和场景规则
        self.system_prompt = (
            "The table numbers are from left to right 1 to 4. "
            "The stall numbers are from left to right 1 to 3. "
            "You are a vision-based event detection assistant for a Drone. "
            "Your job is to analyze the image from the drone's bottom camera and answer the user's questions based on the visual content. IN SHORT, CONCISE ANSWERS ONLY. "
        )

        try:
            response = self.Visionclient.chat.completions.create(
                model="gpt-4.1",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{self.latest_image_b64}"
                                }
                            }
                        ]
                    }
                ]
            )

            answer = response.choices[0].message.content

            # 发布视觉分析结果
            self.get_logger().info(f"VLM Answer: {answer}")
            answer = f"Drone (msg) | {answer}"
            self.pub_input_msg.publish(String(data=answer))
            return True

        except Exception as e:
            self.get_logger().error(f"OpenAI VLM request failed: {e}")
            return False

    # —————————————————————— LLM可调用的无人机动作函数 ——————————————————————

    def describe_screen(self, prompt: str = "What is in front of the drone?"):
        """使用视觉模型描述当前屏幕画面（含状态更新）"""
        self.get_logger().info(f'Describing screen with prompt: "{prompt}"...')
        success = self.query_callback(prompt)
        if success:
            self.get_logger().info(f"Screen description completed.")
            self.robot_task_completed(f"describe screen with prompt: {prompt}")
        else:
            self.robot_task_interrupted(f"describe screen with prompt: {prompt}")
        return

    def hover(self, x: float = 0.0, y: float = 0.0, z: float = 2.0, yaw_deg: float = 0.0):
        """控制无人机悬停到指定位置（含状态更新）"""
        self.get_logger().info(f"Hovering at position x={x}, y={y}, z={z}, yaw={yaw_deg}...")
        success = self.goto_service(x=x, y=y, z=z, yaw_deg=yaw_deg)
        if success:
            self.get_logger().info(f"Hovering at position x={x}, y={y}, z={z} completed.")
            self.robot_task_completed(f"hover at x={x}, y={y}, z={z}")
        else:
            self.robot_task_interrupted(f"hover at x={x}, y={y}, z={z}")
        return

def execute_python_code(code: str, node=None):
    """
    安全地执行LLM生成的Python代码

    参数:
        code: LLM生成的Python代码字符串
        node: RobotLLMNode实例引用

    在受限的命名空间中执行代码，仅暴露必要的对象（node）
    """
    print("Inside the execute python code function")

    if node is None:
        node = RobotLLMNode.get_instance()
        if node is None:
            print("CRITICAL: Could not get node instance!")
            return

    node.get_logger().info(f"Executing generated Python code:{code}")

    try:
        exec(code, {"__builtins__": {}}, {"node": node})
        node.get_logger().info("Code executed successfully")
    except TypeError as e:
        pass
    except Exception as e:
        node.get_logger().error("Failed to execute generated code: %s", e)


def main(args=None):
    """无人机LLM节点入口函数：初始化节点并使用多线程执行器运行"""
    rclpy.init(args=args)
    node = RobotLLMNode()

    executor = rclpy.executors.MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    try:
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
