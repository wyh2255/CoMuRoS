#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
机器人LLM节点模块（Robot LLM Node Module）

该模块实现了基于大语言模型（LLM）的机器人控制节点。RobotLLMNode类作为一个中枢节点，
负责监听聊天主题、解析任务指令、调用OpenAI API生成控制代码，并执行生成的Python代码
来控制机器人（如机械臂）执行拾取等操作。

主要功能：
  - 通过/chat/output监听用户聊天指令
  - 通过/task_manager/tasks_json接收任务管理器下发的任务
  - 调用OpenAI GPT-4o生成对应的机器人控制代码
  - 通过exec执行生成的代码，驱动机器人动作
  - 支持任务取消机制和状态发布
"""

import json
import time

import openai
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from dataclasses import dataclass
import os
from rclpy.callback_groups import (
    MutuallyExclusiveCallbackGroup,
    ReentrantCallbackGroup
)
from ament_index_python.packages import get_package_share_directory
from rclpy.action import ActionClient
from robot_interface.action import PickObject
from robot_interface.srv import StartPick


ROBOT_NAME   = 'robot1'
ROBOT_TYPE   = 'Robotic Arm'
PACKAGE_NAME = 'robot_llm'
NODE_NAME    = 'robot_llm_node'


class TaskCancelledException(Exception):
    """自定义异常，用于信号通知任务已被取消"""
    pass


# 定义可用的机器人动作选项
@dataclass
class TestOption:
    name: str           # 动作名称
    id: int             # 动作唯一标识符
    description: str    # 动作描述，供LLM参考
    example_code: str   # 示例代码，供LLM学习调用方式

option_list = [
    TestOption(
        name="Pick green object",
        id=0,
        description="This is used pick or show the green object or gear",
        example_code="node.pick_green_object()"
    ),
    TestOption(
        name="Pick brown object",
        id=1,
        description="This is used pick or show the brown object or gear",
        example_code="node.pick_brown_object()"
    ),
    TestOption(
        name="Pick grey object",
        id=2,
        description="This is used pick or show the grey object or gear",
        example_code="node.pick_grey_object()"
    )
]

class RobotLLMNode(Node):
    """
    RobotLLM节点 - 基于LLM的机器人控制中枢

    功能概述：
      - 监听聊天话题 (/chat/output, /chat/history, /chat/task_status)
      - 发布任务状态更新 (/chat/task_status 和 <robot_name>_task_status)
      - 接收/处理机器人状态消息（/robot_states，预期为JSON字符串）
      - 调用OpenAI API生成Python控制代码
      - 执行生成的代码以驱动机器人动作
    """

    _instance = None  # 单例模式实例引用

    def __init__(self) -> None:
        """初始化RobotLLMNode节点"""
        super().__init__(NODE_NAME)

        # ---- 参数声明 ----
        self.declare_parameter('robot_name', ROBOT_NAME)
        self.robot_name: str = self.get_parameter('robot_name').value
        robot_task_topic = f'{self.robot_name}_task_status'

        RobotLLMNode._instance = self

        self.current_time = f"Hours: {00}, Minutes: {10}, Seconds: {00}"
        self.robot_task = ""          # 当前机器人任务描述
        self.robot_states = {}        # 所有机器人的状态字典

        # ========== 任务取消机制 ==========
        self._task_cancelled = False  # 简单的布尔取消标志
        # =================================

        # 回调组：用于控制ROS2回调的并发/互斥
        self.single_group = MutuallyExclusiveCallbackGroup()  # 单线程独占回调组
        self.seq_group = MutuallyExclusiveCallbackGroup()     # 顺序执行回调组
        self.multi_group = ReentrantCallbackGroup()           # 可重入回调组（支持并发）

        # ---- 发布者 ----
        self.pub_task_status = self.create_publisher(String, '/chat/task_status', 10)
        self.pub_robot_states = self.create_publisher(String, '/robot_states', 10)
        self.pub_robot_task = self.create_publisher(String, robot_task_topic, 10)

        self._pick_client = self.create_client(StartPick, '/start_pick', callback_group=self.multi_group)
        self._cancel_pub = self.create_publisher(String, '/start_pick/cancel', 10)

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

        # 聊天任务状态订阅（无回调组限制）
        self.sub_chat_task_status = self.create_subscription(
            String, '/chat/task_status', self.on_chat_task_status, 10
        )

        self.get_logger().info(
            f'RobotLLMNode started for robot="{self.robot_name}". '
            f'Publishing robot task status on "{robot_task_topic}".'
        )

        # 设置对话历史和任务历史文件路径
        directry = "data"
        package_path = get_package_share_directory(PACKAGE_NAME)

        script_name = self.robot_name+"_chat_history.txt"
        self.history_file = os.path.join(package_path, directry, script_name)

        script_name = self.robot_name+'_task_history.txt'
        self.robot_task_history = os.path.join(package_path, directry, script_name)

        # 启动时清除旧的历史记录并设置初始状态
        self.clear_files()
        self.robot_has_no_current_task()

    @classmethod
    def get_instance(cls):
        """安全地获取单例节点实例"""
        if cls._instance is None:
            raise RuntimeError("RobotLLMNode has not been created yet! Did you run the node?")
        return cls._instance

    # ========== 取消辅助方法 ==========
    def check_cancelled(self):
        """检查任务是否已被取消，如果已取消则抛出异常"""
        if self._task_cancelled:
            self.get_logger().warn("Task cancellation detected!")
            raise TaskCancelledException("Task was cancelled")

    def reset_cancellation(self):
        """清除取消标志（在开始新任务前调用）"""
        self._task_cancelled = False
    # =================================

    def clear_files(self) -> None:
        """启动时清除对话历史文件和任务历史文件"""
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

    def on_robot_states(self, msg: String) -> None:
        """处理机器人状态更新（msg.data 预期为JSON字符串）"""
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
        """处理来自任务管理器的任务JSON，解析并执行分配给本机器人的任务"""
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

            # 如果第一次没取到，尝试拼接 '_task' 键
            if not robot_task:
                robot_task = robot_tasks.get(self.robot_name+'_task', "").strip()
                self.get_logger().debug(f"{self.robot_name+'_task'} task fetched [2]")

            # 如果还没取到，尝试拼接 '_tasks' 键
            if not robot_task:
                robot_task = robot_tasks.get(self.robot_name+'_tasks', "").strip()
                self.get_logger().debug(f"{self.robot_name+'_tasks'} task fetched [3]")

            # 找不到任务时直接返回
            if not robot_task:
                robot_task = f"No {self.robot_name} task found."
                self.get_logger().debug(robot_task)
                return

            # 如果任务是"stop"指令，则中断所有任务
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

            # ========== 执行新任务前重置取消标志 ==========
            self.reset_cancellation()
            # ==============================================

            self.get_logger().info("Executing Robot Task ..")
            try:
                self.execute_task(robot_task)
                self.get_logger().info(f'On Task Json Task executed')
            except TaskCancelledException:
                self.get_logger().warn("Task execution was cancelled")
                self.robot_task_interrupted(robot_task)

        except json.JSONDecodeError:
            self.get_logger().warn(
                f'Received /task_manager/tasks_json with invalid JSON; raw: {msg.data}'
            )

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


    # -------------------- 任务状态管理方法 --------------------

    def robot_task_in_progress(self, robot_task) -> None:
        """发布任务进行中的状态"""
        if not robot_task:
            robot_task = self.robot_task

        task_in_progress_msg = f"{self.robot_name.capitalize()} (status) : {robot_task} : TASK IN PROGRESS"
        self.get_logger().info(task_in_progress_msg)
        self.robot_task_status_update(task_in_progress_msg)
        return

    def robot_task_completed(self, robot_task=None) -> None:
        """发布任务已完成的状态"""
        if not robot_task:
            robot_task = self.robot_task

        task_completed_msg = f"{self.robot_name.capitalize()} (status) : {robot_task} : TASK COMPLETED"
        self.get_logger().info(task_completed_msg)
        self.robot_task_status_update(task_completed_msg)
        return

    def robot_task_interrupted(self, robot_task=None) -> None:
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

    def tasks_completed(self, task=None) -> None:
        """发布所有任务已完成的状态"""
        msg = String()
        msg.data = f"{self.robot_name.capitalize()} (status): ALL TASKS COMPLETED"
        self.pub_task_status.publish(msg)
        return

    def stop_tasks(self) -> None:
        """停止所有机器人任务：设置取消标志并发送取消信号"""
        self.get_logger().info("Stopping all robot tasks...")

        # ========== 设置取消标志 ==========
        self._task_cancelled = True
        # =================================

        msg = String()
        msg.data = "STOP"
        self._cancel_pub.publish(msg)

        self.get_logger().info("Cancel request sent to /start_pick/cancel")
        self.get_logger().info("All tasks of the robot have been stopped.")

    # -------------------- 核心辅助方法 --------------------

    def read_chat_history(self) -> str:
        """从持久化文件中读取完整的对话历史"""
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
        调用OpenAI GPT-4o生成机器人动作代码

        参数:
            prompt: 系统提示词，描述机器人能力和可用操作
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
                self.get_logger().debug(f"code: {code}")
                self.get_logger().debug(f"explaination: {explaination}")
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
        执行给定的机器人任务：通过LLM生成控制代码并执行

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
                [f"Function Name: {opt.name} \nFunction Description: {opt.description} (e.g., {opt.example_code})"
                 for opt in option_list]
            )

            self.get_logger().debug(f"Action message: {available_actions}")
            self.get_logger().info("Building system messages")

            # 构建LLM提示词，包含机器人信息、对话历史、状态和可用操作
            prompt = (
                f"You are a robot control system controlling a {ROBOT_TYPE} named '{self.robot_name}'. "
                "You can generate python code to perform actions. "
                "Based on the given task, you need to choose the appropriate action from the available options. "
                f"Recent Tasks (History): {chat_history} "
                f"Current States of All Robots: {self.robot_states} "
                f"Available Actions: {available_actions} "
                "Using the class reference name same as the example is important. "
                "Use the name 'node' to refer to the RobotLLMNode instance. "
                "Your generating codes are case-sensitive, so DO NOT change the case of any function or variable names. "
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

    def pick_object(self, object_name: str = "red_gear") -> bool:
        """
        调用StartPick服务拾取指定物体（异步轮询等待，不嵌套spin）

        参数:
            object_name: 要拾取的物体名称

        返回:
            bool: 拾取是否成功
        """
        # 执行前检查是否已被取消
        self.check_cancelled()

        self.get_logger().info(f"Picking '{object_name}'...")

        if not self._pick_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("StartPick service not available!")
            return False

        req = StartPick.Request()
        req.object_name = object_name

        future = self._pick_client.call_async(req)
        self.get_logger().info(f"Waiting for /start_pick response for '{object_name}'...")

        deadline = time.time() + 1200000.0
        while rclpy.ok() and not future.done():
            # 在等待循环中也检查取消状态
            self.check_cancelled()
            time.sleep(0.05)
            if time.time() > deadline:
                self.get_logger().error(f"Service call for '{object_name}' timed out.")
                return False

        try:
            res = future.result()
        except Exception as e:
            self.get_logger().error(f"Service call failed: {e}")
            return False

        if res.success:
            self.get_logger().info(f"Pick succeeded: {res.message}")
            return True
        else:
            self.get_logger().warn(f"Pick failed: {res.message}")
            return False

    # —————————————————————— LLM可调用的机器人动作函数 ——————————————————————

    def pick_green_object(self) -> bool:
        """拾取绿色物体（含取消支持和状态更新）"""
        self.get_logger().info("Picking green object...")
        success = self.pick_object("green object")
        if success:
            self.robot_task_completed("pick green object")
        else:
            self.robot_task_interrupted("pick green object")
        return success

    def pick_brown_object(self) -> bool:
        """拾取棕色物体（含取消支持和状态更新）"""
        self.get_logger().info("Picking brown object...")
        success = self.pick_object("brown object")
        if success:
            self.robot_task_completed("pick brown object")
        else:
            self.robot_task_interrupted("pick brown object")
        return success

    def pick_grey_object(self) -> bool:
        """拾取灰色物体（含取消支持和状态更新）"""
        self.get_logger().info("Picking grey object...")
        success = self.pick_object("grey object")
        if success:
            self.robot_task_completed("pick grey object")
        else:
            self.robot_task_interrupted("pick grey object")
        return success


def execute_python_code(code: str, node=None):
    """
    安全地执行LLM生成的Python代码

    参数:
        code: LLM生成的Python代码字符串
        node: RobotLLMNode实例引用

    在受限的命名空间中执行代码，仅暴露必要的对象（node, TaskCancelledException）
    """
    print("Inside the execute python code function")

    if node is None:
        node = RobotLLMNode.get_instance()
        if node is None:
            print("CRITICAL: Could not get node instance!")
            return

    node.get_logger().info(f"Executing generated Python code: {code}")

    try:
        # ========== 将TaskCancelledException传递给exec上下文 ==========
        exec(code, {"__builtins__": {}}, {
            "node": node,
            "TaskCancelledException": TaskCancelledException
        })
        # ==============================================================
        node.get_logger().info("Code executed successfully")

    except TaskCancelledException:
        node.get_logger().warn("Code execution cancelled")
        raise  # 重新抛出，让execute_task处理
    except TypeError as e:
        pass
    except Exception as e:
        node.get_logger().error("Failed to execute generated code: %s", e)


def main(args=None):
    """节点入口函数：初始化节点并使用多线程执行器运行"""
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