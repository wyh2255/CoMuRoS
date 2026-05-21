#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
雅虎（Yahboom）全向机器人LLM控制节点模块

该模块实现了基于大语言模型（LLM）的全向移动机器人（Yahboom Rosmaster X3）控制节点。
RobotLLMNode类作为一个中枢节点，负责：
  - 监听聊天主题，接收任务指令
  - 调用OpenAI GPT-4o生成控制代码
  - 执行生成的Python代码控制机器人移动和清洁
  - 发布任务状态更新

主要功能：
  - 全向移动：通过goto_service控制机器人移动到指定(x, y, yaw)
  - 清洁任务：沿预设路径自动执行清洁
  - 障碍物移除：通过Ignition服务移除仿真中的障碍物
  - 任务取消机制
"""

import json
import time
import subprocess

import openai
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from dataclasses import dataclass
import os
from rclpy.callback_groups import (
    MutuallyExclusiveCallbackGroup,
    ReentrantCallbackGroup
)
from ament_index_python.packages import get_package_share_directory

from robot_interface.srv import GotoPoseHolonomic



ROBOT_NAME   = 'cleaning_bot'
ROBOT_TYPE   = 'Holonomic Drive Robot'
NODE_NAME    = 'cleaning_bot_llm_node'
PACKAGE_NAME = 'yahboom_llm'

# 定义全向机器人可用的动作选项
@dataclass
class TestOption:
    name: str           # 动作名称
    id: int             # 动作唯一标识
    description: str    # 动作描述，供LLM参考
    example_code: str   # 示例调用代码

option_list = [
    TestOption(
        name='Clean',
        id=0,
        description='Navigate along the predefined cleaning path to clean the area.',
        example_code="node.clean()"
    )
]


class RobotLLMNode(Node):
    """
    全向机器人LLM控制节点

    功能概述：
      - 监听聊天话题（/chat/output等），接收用户任务指令
      - 发布任务状态更新（/chat/task_status 和 <robot_name>_task_status）
      - 接收/处理机器人状态消息（/robot_states，JSON格式）
      - 调用OpenAI API生成Python控制代码
      - 支持全向移动（goto_service）和清洁任务
      - 支持任务取消机制
    """

    _instance = None  # 单例实例引用

    def __init__(self) -> None:
        """初始化全向机器人LLM控制节点"""
        super().__init__(NODE_NAME)

        # ---- 参数声明 ----
        self.declare_parameter('robot_name', ROBOT_NAME)
        self.robot_name: str = self.get_parameter('robot_name').value
        robot_task_topic = f'{self.robot_name}_task_status'

        RobotLLMNode._instance = self

        self.current_time = f"Hours: {00}, Minutes: {10}, Seconds: {00}"
        self.robot_task = ""           # 当前任务描述
        self.robot_states = {}         # 机器人状态字典

        # 回调组配置
        self.single_group = MutuallyExclusiveCallbackGroup()  # 互斥回调组（聊天等独占操作）
        self.seq_group = MutuallyExclusiveCallbackGroup()     # 顺序执行回调组
        self.multi_group = ReentrantCallbackGroup()           # 可重入回调组（支持并发）

        # ---- 发布者 ----
        self.pub_task_status = self.create_publisher(String, '/chat/task_status', 10)
        self.pub_robot_states = self.create_publisher(String, '/robot_states', 10)
        self.pub_robot_task = self.create_publisher(String, robot_task_topic, 10)

        # 服务客户端
        self._goto_client = self.create_client(GotoPoseHolonomic, "/r1/goto_pose", callback_group=self.multi_group)
        self._find_client = self.create_client(Find, '/find', callback_group=self.multi_group)

        # 取消指令发布者
        self._cancel_goto_pub = self.create_publisher(Bool, "/r1/cancel_goto_pose_goal", 10)
        self._cancel_find_pub = self.create_publisher(Bool, '/find/cancel', 10)

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
            if all(isinstance(v, dict) for v in self.robot_states.values()) and len(self.robot_states) > 0:
                self.robot_states = {"robot_states": self.robot_states}
                self.get_logger().debug("Wrapped existing dict under 'robot_states'")
            else:
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
        self._cancel_find_pub.publish(msg)
        self._cancel_goto_pub.publish(msg)
        self.get_logger().info("Cancel request sent to /goto/cancel and /find/cancel")
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
        调用OpenAI GPT-4o生成机器人控制代码

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
                messages=[{'role': 'system', 'content': prompt}, {'role': 'user', 'content': task}],
                max_tokens=500,
                temperature=0.5,
            )
            self.get_logger().debug(f"LLM Response: {response}")
            raw = response.choices[0].message.content.strip()
            self.get_logger().info(f"Content: {raw}")
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
            available_actions = "\n".join(
                [f"Function Name: {opt.name} \nFunction Description: {opt.description} (e.g., {opt.example_code})" for opt in option_list]
            )
            self.get_logger().debug(f"Action message: {available_actions}")
            self.get_logger().info("Building system messages")
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

    def find_service(self, name: str = "red_gear") -> bool:
        """调用Find服务查找物体（非阻塞，无嵌套spin）"""
        self.get_logger().info(f"Finding '{name}'...")
        if not self._find_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("Find service not available!")
            return False
        req = Find.Request()
        req.name = name
        future = self._find_client.call_async(req)
        self.get_logger().info(f"Waiting for /find response for '{name}'...")
        deadline = time.time() + 120.0
        while rclpy.ok() and not future.done():
            time.sleep(0.05)
            if time.time() > deadline:
                self.get_logger().error(f"Service call for Find '{name}' timed out.")
                return False
        try:
            res = future.result()
        except Exception as e:
            self.get_logger().error(f"Service call failed: {e}")
            return False
        if res.success:
            self.get_logger().info(f"Find succeeded: {res.message}")
            return True
        else:
            self.get_logger().warn(f"Find failed: {res.message}")
            return False

    def goto_service(self, x: float, y: float, yaw_deg: float) -> bool:
        """
        调用全向机器人GoTo位置控制服务（非阻塞，无嵌套spin）

        参数:
            x, y: 目标位置坐标（米）
            yaw_deg: 目标偏航角（度）

        返回:
            bool: 移动是否成功
        """
        self.get_logger().info(f"Holonomic robot moving to x={x}, y={y}, yaw={yaw_deg}°...")
        if not self._goto_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("GotoPoseHolonomic service NOT available!")
            return False
        req = GotoPoseHolonomic.Request()
        req.x = x
        req.y = y
        req.yaw_deg = yaw_deg
        future = self._goto_client.call_async(req)
        self.get_logger().info("Waiting for holonomic service response...")
        deadline = time.time() + 1000000.0
        while rclpy.ok() and not future.done():
            time.sleep(0.05)
            if time.time() > deadline:
                self.get_logger().error("Holonomic goto service call TIMED OUT!")
                return False
        try:
            res = future.result()
        except Exception as e:
            self.get_logger().error(f"Holonomic goto service call FAILED: {e}")
            return False
        if not res.accepted:
            self.get_logger().warn(f"Holonomic goto NOT accepted: {res.message}")
            return False
        if res.success:
            self.get_logger().info(f"Holonomic goto SUCCESS: {res.message}")
            return True
        self.get_logger().warn(f"Holonomic goto FAILED: {res.message}")
        return False

    # —————————————————————— LLM可调用的机器人动作函数 ——————————————————————

    def remove_cube(self):
        """
        通过Ignition服务移除仿真环境中的小方块障碍物

        调用Ignition的remove服务来移除名为"small_cube"的实体。
        """
        cmd = [
            'ign', 'service', '-s', '/world/food_court/remove',
            '--reqtype', 'ignition.msgs.Entity',
            '--reptype', 'ignition.msgs.Boolean',
            '--timeout', '1000',
            '--req', 'name: "small_cube", type: 2'
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(f"Remove result: {result.stdout}")
        return result.returncode == 0

    def clean(self) -> None:
        """
        执行清洁任务：沿预设路径导航并在指定位置执行清洁操作

        清洁路径包含多个航点，机器人依次遍历各航点。
        在第二个航点处尝试移除障碍物（small_cube）。
        """
        self.get_logger().info(f"Starting cleaning task...")

        # 清洁路径航点列表（x, y, yaw）
        cleaning_locations = [
            (11.0, -3.0, 0.0),
            (4.0, -3.0, 0.0),
            (-3.5, -3.0, 0.0),
            (-3.5,  3.0, 0.0),
            (11.0,  3.0, 0.0),
            (11.0,  0.0, 0.0)
        ]

        for idx, (x, y, yaw) in enumerate(cleaning_locations):
            self.get_logger().info(f"Navigating to cleaning location {idx + 1} at x={x}, y={y}, yaw={yaw}°")
            success = self.goto_service(x=x, y=y, yaw_deg=yaw)
            if not success:
                self.get_logger().error(f"Failed to reach cleaning location {idx + 1}. Aborting cleaning task.")
                self.robot_task_interrupted("clean")
                return

            self.get_logger().info(f"Performing cleaning at location {idx + 1}...")
            # 在第二个航点移除障碍物
            if idx == 1:
                self.get_logger().info("Removing obstacle (small_cube) during cleaning...")
                remove_success = self.remove_cube()
                if remove_success:
                    self.get_logger().info("Obstacle removed successfully.")
                else:
                    self.get_logger().warn("Failed to remove obstacle.")
            time.sleep(2)

        self.get_logger().info("Cleaning task completed successfully.")
        self.robot_task_completed("clean")
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
    """全向机器人LLM节点入口函数：初始化节点并使用多线程执行器运行"""
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
