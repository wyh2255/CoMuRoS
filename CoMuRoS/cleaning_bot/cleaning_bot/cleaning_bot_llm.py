#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
清洁机器人LLM节点模块

该模块实现了清洁机器人（cleaning_bot）的核心控制节点，通过大语言模型(LLM)进行
任务决策和代码生成。机器人采用全向驱动（Holonomic Drive）方式，负责在餐厅环境中
执行清洁任务。

主要功能：
  - 订阅聊天话题（/chat/output, /chat/history, /chat/task_status）
  - 发布任务状态更新（/chat/task_status 和机器人特定话题）
  - 监听机器人状态消息（/robot_states，期望JSON格式）
  - 通过OpenAI GPT-4o生成任务执行代码
  - 支持任务取消机制，可中断正在执行的任务
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



# ==================== 机器人配置常量 ====================
ROBOT_NAME   = 'cleaning_bot'          # 机器人名称
ROBOT_TYPE   = 'Holonomic Drive Robot' # 机器人类型：全向驱动
NODE_NAME    = 'cleaning_bot_llm_node' # ROS2节点名称
PACKAGE_NAME = 'cleaning_bot'          # ROS2包名称


class TaskCancelledException(Exception):
    """自定义异常：用于信号通知任务被取消"""
    pass


# ==================== 定义可用动作 ====================
@dataclass
class TestOption:
    """测试选项数据结构，定义机器人可执行的一个动作

    属性:
        name: 动作名称
        id: 动作唯一标识符
        description: 动作描述
        example_code: 调用示例代码
    """
    name: str
    id: int
    description: str
    example_code: str

option_list = [
    TestOption(
        name='Clean',
        id=0,
        description='沿预定义的清洁路径导航以清洁区域。',
        example_code="node.clean()"
    )
]


class RobotLLMNode(Node):
    """
    机器人LLM核心节点（单例模式）

    该节点作为机器人的"大脑"，通过LLM进行任务决策：
      - 监听聊天相关话题（/chat/output, /chat/history, /chat/task_status）
      - 发布任务状态更新（/chat/task_status）和机器人特定状态（<robot_name>_task_status）
      - 镜像/消费 /robot_states 话题上的机器人状态消息（JSON格式字符串）
      - 支持通过OpenAI API动态生成和执行Python控制代码
      - 实现任务取消机制，允许安全中断正在执行的任务
    """

    _instance = None

    def __init__(self) -> None:
        """初始化机器人LLM节点，设置参数、发布器、订阅器和回调组"""
        super().__init__(NODE_NAME)

        # ---- 参数声明与获取 ----
        self.declare_parameter('robot_name', ROBOT_NAME)
        self.robot_name: str = self.get_parameter('robot_name').value
        robot_task_topic = f'{self.robot_name}_task_status'

        # 单例实例赋值
        RobotLLMNode._instance = self

        # 初始状态变量
        self.current_time = f"Hours: {00}, Minutes: {10}, Seconds: {00}"
        self.robot_task = ""          # 当前机器人任务描述
        self.robot_states = {}        # 所有机器人的状态字典

        # ========== 任务取消机制（标志位）==========
        self._task_cancelled = False  # 简单的布尔标志，用于信号任务取消
        # ============================================

        # ========== 回调组 ==========
        # MutuallyExclusiveCallbackGroup: 互斥回调组，同一组内回调互斥执行
        # ReentrantCallbackGroup: 可重入回调组，允许并发执行
        self.single_group = MutuallyExclusiveCallbackGroup()
        self.seq_group = MutuallyExclusiveCallbackGroup()
        self.multi_group = ReentrantCallbackGroup()

        # ---- 发布器（Publishers）----
        self.pub_task_status = self.create_publisher(String, '/chat/task_status', 10)
        self.pub_robot_states = self.create_publisher(String, '/robot_states', 10)
        self.pub_robot_task = self.create_publisher(String, robot_task_topic, 10)

        # GoTo服务客户端（用于调用全向移动控制器的导航服务）
        self._goto_client = self.create_client(GotoPoseHolonomic, "/r1/goto_pose", callback_group=self.multi_group)
        # 取消导航目标发布器
        self._cancel_goto_pub = self.create_publisher(Bool, "/r1/cancel_goto_pose_goal", 10)

        # ---- 订阅器（Subscriptions）----
        # 订阅所有机器人的状态话题
        self.sub_robot_states = self.create_subscription(
            String, '/robot_states', self.on_robot_states, 10,
            callback_group=self.seq_group
        )
        # 订阅当前时间话题
        self.current_time_sub = self.create_subscription(
            String, '/current_time', self.on_current_time, 10,
            callback_group=self.seq_group
        )
        # 订阅任务管理器下发的任务JSON话题
        self.sub_tasks_json = self.create_subscription(
            String, '/task_manager/tasks_json', self.on_tasks_json, 10,
            callback_group=self.multi_group
        )
        # 订阅聊天输出话题
        self.sub_chat_output = self.create_subscription(
            String, '/chat/output', self.on_chat_output, 10,
            callback_group=self.single_group
        )

        # self.sub_chat_history = self.create_subscription(
        #     String, '/chat/history', self.on_chat_history, 10
        # )
        # 订阅聊天任务状态话题
        self.sub_chat_task_status = self.create_subscription(
            String, '/chat/task_status', self.on_chat_task_status, 10
        )

        # ---- 定时器回调 (已注释掉) ----
        # self.timer_period = 1.0  # seconds
        # self.robot_task_status_callback = self.create_timer(
        #     self.timer_period, self.robot_task_status_update,
        #     callback_group=self.multi_group
        # )

        self.get_logger().info(
            f'RobotLLMNode started for robot="{self.robot_name}". '
            f'Publishing robot task status on "{robot_task_topic}".'
        )

        # ========== 文件路径初始化 ==========
        directry = "data"
        package_path = get_package_share_directory(PACKAGE_NAME)

        # 聊天历史记录文件
        script_name = self.robot_name+"_chat_history.txt"
        self.history_file = os.path.join(package_path, directry, script_name)

        # 机器人任务历史记录文件
        script_name = self.robot_name+'_task_history.txt'
        self.robot_task_history = os.path.join(package_path, directry, script_name)

        # 启动时清理旧文件并初始化任务状态为"无任务"
        self.clear_files()
        self.robot_has_no_current_task()

    @classmethod
    def get_instance(cls):
        """安全获取或创建单例节点实例"""
        if cls._instance is None:
            raise RuntimeError("RobotLLMNode has not been created yet! Did you run the node?")
        return cls._instance

    # ========== 取消辅助方法 ==========
    def check_cancelled(self):
        """检查任务是否已被取消。如果已取消则抛出异常。"""
        if self._task_cancelled:
            self.get_logger().warn("Task cancellation detected!")
            raise TaskCancelledException("Task was cancelled")

    def reset_cancellation(self):
        """清除取消标志（在开始新任务前调用）"""
        self._task_cancelled = False
    # ==========================================

    def clear_files(self) -> None:
        """启动时清除聊天历史文件和机器人任务历史文件"""
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

    # -------------------- Callbacks --------------------

    # ==================== 话题回调函数 ====================

    def on_robot_states(self, msg: String) -> None:
        """
        /robot_states 话题回调函数
        处理所有机器人的状态更新消息（msg.data 包含JSON格式字符串）
        """
        try:
            data = json.loads(msg.data)
            rs = data.get("robot_states")
            if isinstance(rs, dict):
                self.robot_states = rs  # 更新本地存储的机器人状态字典
                self.get_logger().debug(f"robot_states keys: {list(self.robot_states.keys())}")
            else:
                self.get_logger().warning("/robot_states missing or not a dict; ignoring payload.")
        except json.JSONDecodeError:
            self.get_logger().error(f"/robot_states not JSON: {msg.data}")

    def on_tasks_json(self, msg: String) -> None:
        """
        /task_manager/tasks_json 话题回调函数
        处理任务管理器下发的任务JSON，解析并执行当前机器人的任务

        任务键名匹配优先级:
          robot_name > robot_name+'_task' > robot_name+'_tasks'
        """
        self.get_logger().debug(f'Received /task_manager/tasks_json: {msg.data}')
        try:
            data = json.loads(msg.data)
            # 标准化任务键名：转为小写并将空格替换为下划线
            robot_tasks = {
                key.lower().replace(" ", "_"): value
                for key, value in data.get("robot_tasks", {}).items()
            }

            # 尝试匹配当前机器人对应的任务键名
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

            # 如果任务包含"stop"关键字，则执行停止操作
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

            # ========== 开始新任务前重置取消标志 ==========
            self.reset_cancellation()
            # =========================================================

            self.get_logger().info("Executing Robot Task ..")
            try:
                self.execute_task(robot_task)
                self.get_logger().info(f'On Task Json Task executed')
            except TaskCancelledException:
                # 捕获取消异常并更新任务状态为已中断
                self.get_logger().warn("Task execution was cancelled")
                self.robot_task_interrupted(robot_task)

        except json.JSONDecodeError:
            self.get_logger().warn(
                f'Received /task_manager/tasks_json with invalid JSON; raw: {msg.data}'
            )

    def on_chat_output(self, msg: String) -> None:
        """
        /chat/output 话题回调函数
        处理原始聊天输出并将其保存到历史记录文件中
        """
        self.get_logger().debug(f'Received /chat/output: {msg.data}')

        timestamp = self.current_time
        parts = msg.data.split("|", 1)

        if len(parts) == 2:
            role, content = parts[0].strip(), parts[1].strip()
            self.chat_entry = f"[Time: {timestamp}] {role.capitalize()}: {content}"
        else:
            role, content = "Task Manager", msg.data
            self.chat_entry = f"[Time: {timestamp}] {role}:\n{content}"

        with open(self.history_file, "a") as file:
            file.write(self.chat_entry + "\n")

    def on_chat_task_status(self, msg: String) -> None:
        """
        /chat/task_status 话题回调函数
        监听外部任务状态变化并记录到历史文件
        """
        self.get_logger().debug(f'Observed /chat/task_status: {msg.data}')
        with open(self.history_file, "a") as file:
            file.write(msg.data + "\n")

    def on_current_time(self, msg: String) -> None:
        """
        /current_time 话题回调函数
        更新时间信息（从模拟时间话题获取）
        """
        self.get_logger().debug(f'Received /current_time: {msg.data}')
        self.current_time = msg.data

    # ==================== 状态管理辅助函数 ====================

    def update_robot_state(self, incoming: dict, robot_name: str | None = None) -> None:
        """
        将部分机器人状态更新合并到 self.robot_states 中

        处理逻辑：
        - self.robot_states 可能是：None、无"robot_states"键的字典、已有"robot_states"键的字典、或旧格式的裸字典
        - 确保顶层"robot_states"键存在
        - 确保每个机器人的条目存在
        - 添加新键并更新已更改的键
        - 最后发布更新后的完整状态到/robot_states话题
        """
        # 如果未指定机器人名称，默认使用当前节点管理的机器人名称
        if robot_name is None:
            robot_name = self.robot_name

        # 1) 确保顶层容器存在
        if self.robot_states is None:
            self.robot_states = {}
            self.get_logger().info("Created top-level robot_states container dict")

        # 2) 确保顶层存在"robot_states"键
        if "robot_states" not in self.robot_states:
            # 启发式判断：如果当前字典看起来只包含机器人条目，则将其包装到"robot_states"下
            if all(isinstance(v, dict) for v in self.robot_states.values()) and len(self.robot_states) > 0:
                self.robot_states = {"robot_states": self.robot_states}
                self.get_logger().debug("Wrapped existing dict under 'robot_states'")
            else:
                # 创建新的robot_states字典，保留其他元数据字段
                self.robot_states["robot_states"] = {}
                self.get_logger().info("Created 'robot_states' key in top-level dict")

        robots_dict = self.robot_states["robot_states"]

        # 3) 确保机器人条目存在
        if robot_name not in robots_dict or not isinstance(robots_dict[robot_name], dict):
            robots_dict[robot_name] = {}
            self.get_logger().info(f"Created new robot entry '{robot_name}'")

        saved_state = robots_dict[robot_name]

        # 4) 将传入的键值对合并到该机器人的状态中
        for key, new_val in incoming.items():
            if key not in saved_state:
                saved_state[key] = new_val
                self.get_logger().debug(f"{robot_name}: Added '{key}' = '{new_val}'")
            else:
                old_val = saved_state[key]
                if old_val != new_val:
                    saved_state[key] = new_val
                    self.get_logger().debug(f"{robot_name}: Updated '{key}' from '{old_val}' → '{new_val}'")

        # 5) 发布更新后的机器人状态到/robot_states话题
        msg = String()
        msg.data = json.dumps(self.robot_states)
        self.pub_robot_states.publish(msg)

        # 6) 记录更新后的状态结构
        self.get_logger().debug(f"robot_states now: {self.robot_states['robot_states']}")
        return

    # ==================== 任务状态管理方法 ====================

    def robot_task_in_progress(self, robot_task) -> None:
        """标记任务为"进行中"状态并发布更新"""
        if not robot_task:
            robot_task = self.robot_task

        task_in_progress_msg = f"{self.robot_name.capitalize()} (status) : {robot_task} : TASK IN PROGRESS"
        self.get_logger().info(task_in_progress_msg)
        self.robot_task_status_update(task_in_progress_msg)
        return

    def robot_task_completed(self, robot_task) -> None:
        """标记任务为"已完成"状态并发布更新"""
        if not robot_task:
            robot_task = self.robot_task

        task_completed_msg = f"{self.robot_name.capitalize()} (status) : {robot_task} : TASK COMPLETED"
        self.get_logger().info(task_completed_msg)
        self.robot_task_status_update(task_completed_msg)
        return

    def robot_task_interrupted(self, robot_task) -> None:
        """标记任务为"已中断"状态并发布更新"""
        if not robot_task:
            robot_task = self.robot_task

        task_interrupted_msg = f"{self.robot_name.capitalize()} (status) : {robot_task} : TASK INTERRUPTED"
        self.get_logger().info(task_interrupted_msg)
        self.robot_task_status_update(task_interrupted_msg)
        return

    def robot_has_no_current_task(self) -> None:
        """标记当前无任务状态并发布更新"""
        robot_task = ""
        no_task_msg = f"{self.robot_name.capitalize()} (status) : {robot_task} : NO CURRENT TASK"
        self.get_logger().info(no_task_msg)
        self.robot_task_status_update(no_task_msg)
        return

    def robot_task_status_update(self, status_msg: str) -> None:
        """
        发布机器人任务状态更新

        发布到机器人特定任务状态话题，并同时写入任务历史文件
        """
        msg = String()
        msg.data = status_msg

        # 发布到机器人特定话题（如 cleaning_bot_task_status）
        self.pub_robot_task.publish(msg)

        try:
            with open(self.robot_task_history, "a") as file:
                file.write(f"{status_msg}\n")
        except FileNotFoundError as e:
            self.get_logger().warn(f"Robot Task history file not found: {e}")

        self.get_logger().info("Robot task status updated... ")
        return

    def tasks_completed(self, task) -> None:
        """发布所有任务已完成的状态到/chat/task_status话题"""
        msg = String()
        msg.data = f"{self.robot_name.capitalize()} (status): ALL TASKS COMPLETED"
        self.pub_task_status.publish(msg)
        return

    def stop_tasks(self) -> None:
        """
        停止所有机器人任务

        通过设置取消标志和向取消话题发布消息来中断任务执行
        """
        self.get_logger().info("Stopping all robot tasks...")

        # ========== 设置取消标志 ==========
        self._task_cancelled = True
        # ===========================================

        # 向全向移动控制器发布取消目标消息
        msg = Bool()
        msg.data = True
        self._cancel_goto_pub.publish(msg)

        self.get_logger().info("Cancel request sent to /goto/cancel")
        self.get_logger().info("Cancel request sent to /find/cancel")
        self.get_logger().info("All tasks of the robot have been stopped.")

    # ==================== 辅助方法 ====================

    def read_chat_history(self) -> str:
        """
        从持久化文件中读取完整的聊天历史记录

        包含完善的错误处理（文件不存在、权限错误、OS错误等）
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
        通过OpenAI GPT-4o生成动作代码

        发送系统提示词和用户任务给LLM，解析返回的Python代码

        返回:
            (code, explanation): 生成的Python代码和解释说明
            失败时返回 (None, None)
        """
        self.get_logger().info(f"Generating code...")
        from openai import OpenAI
        try:
            client = OpenAI(
            api_key=os.environ.get('DEEPSEEK_API_KEY'),
            base_url="https://api.deepseek.com")
            response = client.chat.completions.create(
                model="deepseek-v4-flash",
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

            # 解析三引号包裹的Python代码块
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
        使用LLM决策执行指定的机器人任务

        流程:
        1. 读取聊天历史
        2. 构建系统提示词（包含可用动作、机器人状态等信息）
        3. 调用OpenAI生成执行代码
        4. 执行生成的代码
        5. 更新任务完成状态
        """
        chat_history = self.read_chat_history()

        try:
            self.get_logger().info("Building system action messages")
            # 构建可用动作描述列表
            available_actions = "\n".join(
                [f"Function Name: {opt.name} \nFunction Description: {opt.description} (e.g., {opt.example_code})"
                 for opt in option_list]
            )

            self.get_logger().debug(f"Action message: {available_actions}")
            self.get_logger().info("Building system messages")

            # 构建发送给LLM的系统提示词
            prompt = (
                f"You are a robot control system controlling a {ROBOT_TYPE} named '{self.robot_name}'. "
                "You must generate python code to perform the task. "
                "Based on the given task, generate code using available actions."
                f"Recent Tasks (History): {chat_history} "
                f"Current States of All Robots: {self.robot_states} "
                f"Available Actions: {available_actions} "
                "Using the class reference name same as the example is important. "
                "Use the name 'node' to refer to the RobotLLMNode instance. "
            )

            self.get_logger().debug(f"Prompt message: {prompt}")

        except Exception as e:
            self.get_logger().warning(f"{e}")

        # 调用LLM生成代码
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

    def goto_service(self, x: float, y: float, yaw_deg: float) -> bool:
        """
        调用全向移动控制器的GoTo服务并等待结果

        非阻塞方式调用（不嵌套spin），在等待循环中支持任务取消检测

        参数:
            x: 目标X坐标
            y: 目标Y坐标
            yaw_deg: 目标偏航角（度）

        返回:
            bool: 是否成功到达目标点
        """
        # ========== 检查取消 ==========
        self.check_cancelled()
        # ========================================

        self.get_logger().info(f"Holonomic robot moving to x={x}, y={y}, yaw={yaw_deg}°...")

        if not self._goto_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("GotoPoseHolonomic service NOT available!")
            return False

        # 构建GoTo服务请求
        req = GotoPoseHolonomic.Request()
        req.x = x
        req.y = y
        req.yaw_deg = yaw_deg

        # 异步调用服务
        future = self._goto_client.call_async(req)
        self.get_logger().info("Waiting for holonomic service response...")

        # 在等待循环中检查取消标志
        deadline = time.time() + 1000000.0
        while rclpy.ok() and not future.done():
            # ========== 循环中检查取消 ==========
            self.check_cancelled()
            # ================================================
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

    def remove_cube(self):
        """
        通过Ignition Gazebo服务移除障碍物方块

        调用 /world/food_court/remove 服务移除名为"small_cube"的实体
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
        执行餐厅清洁任务

        沿预定义的6个清洁点依次导航，在每个位置执行清洁操作。
        支持任务取消检测（在每次移动前检查）和状态更新发布。
        在位置2（索引1）处特殊处理障碍物（small_cube）移除。

        清洁路径:
          1. (11.0, -3.0) -> 2. (4.0, -3.0) -> 3. (-3.5, -3.0)
          4. (-3.5, 3.0) -> 5. (11.0, 3.0) -> 6. (11.0, 0.0)
        """
        self.get_logger().info(f"Starting restaurant cleaning task...")

        # 更新状态：任务开始
        self.update_robot_state({
            "current_task": "clean_restaurant",
            "task_status": "in_progress",
            "cleaning_progress": "0/6",
            "cleaning_description": "Cleaning restaurant floor areas"
        })

        # 预定义的6个清洁点坐标
        cleaning_locations = [
            (11.0, -3.0, 0.0),
            (4.0, -3.0, 0.0),
            (-3.5, -3.0, 0.0),
            (-3.5, 3.0, 0.0),
            (11.0, 3.0, 0.0),
            (11.0, 0.0, 0.0)
        ]

        for idx, (x, y, yaw) in enumerate(cleaning_locations):
            # ========== 每次移动前检查取消 ==========
            self.check_cancelled()
            # ========================================

            # 更新状态：正在导航到清洁点
            self.update_robot_state({
                "task_status": "navigating",
                "target_location": f"cleaning_point_{idx + 1}",
                "target_coords": {"x": x, "y": y, "yaw": yaw},
                "cleaning_progress": f"{idx}/6"
            })

            self.get_logger().info(f"Navigating to cleaning location {idx + 1} at x={x}, y={y}, yaw={yaw}°")
            success = self.goto_service(x=x, y=y, yaw_deg=yaw)

            if not success:
                self.get_logger().error(f"Failed to reach cleaning location {idx + 1}. Aborting cleaning task.")

                # 更新状态：导航失败
                self.update_robot_state({
                    "task_status": "failed",
                    "failure_reason": f"navigation_failed_at_location_{idx + 1}",
                    "cleaning_progress": f"{idx}/6"
                })

                self.robot_task_interrupted("clean")
                return

            # 更新状态：到达清洁点，开始清洁
            self.update_robot_state({
                "task_status": "cleaning_area",
                "current_location": f"cleaning_point_{idx + 1}",
                "current_coords": {"x": x, "y": y, "yaw": yaw},
                "activity": "Cleaning restaurant floor"
            })

            self.get_logger().info(f"Cleaning restaurant area at location {idx + 1}...")

            # 特殊处理：位置2（索引1）遇到障碍物
            if idx == 1:
                self.get_logger().info("Obstacle (small_cube) detected during cleaning - attempting removal...")

                # 更新状态：正在移除障碍物
                self.update_robot_state({
                    "task_status": "removing_obstacle",
                    "obstacle_type": "small_cube",
                    "activity": "Removing obstacle blocking cleaning path"
                })

                remove_success = self.remove_cube()

                if remove_success:
                    self.get_logger().info("Obstacle removed successfully. Continuing restaurant cleaning.")

                    # 更新状态：障碍物已移除，继续清洁
                    self.update_robot_state({
                        "obstacle_removed": True,
                        "obstacle_removal_location": f"cleaning_point_{idx + 1}",
                        "obstacle_removal_status": "Success",
                        "activity": "Resumed cleaning after obstacle removal"
                    })
                else:
                    self.get_logger().warn("Failed to remove obstacle. Continuing with remaining cleaning areas.")

                    # 更新状态：障碍物移除失败，继续
                    self.update_robot_state({
                        "obstacle_removed": False,
                        "obstacle_removal_failed": True,
                        "obstacle_removal_status": "Failed",
                        "activity": "Continuing cleaning (obstacle removal failed)"
                    })

            # 模拟清洁时间
            self.get_logger().info(f"Cleaning restaurant floor at location {idx + 1}...")
            time.sleep(2.0)

            # 更新状态：当前位置清洁完成
            self.update_robot_state({
                "task_status": "in_progress",
                "cleaning_progress": f"{idx + 1}/6",
                f"location_{idx + 1}_cleaned": True,
                f"location_{idx + 1}_cleaned_at": time.time(),
                "activity": f"Completed cleaning at point {idx + 1}"
            })

            self.get_logger().info(f"Restaurant area {idx + 1} cleaned successfully.")

        # 更新状态：所有清洁任务完成
        self.update_robot_state({
            "task_status": "completed",
            "cleaning_progress": "6/6",
            "all_locations_cleaned": True,
            "completion_timestamp": time.time(),
            "activity": "Restaurant cleaning completed successfully"
        })

        self.get_logger().info("Restaurant cleaning task completed successfully.")
        self.robot_task_completed("clean")
        return


def execute_python_code(code: str, node=None):
    """
    安全执行由LLM生成的Python代码

    在受限的exec上下文中执行代码，仅暴露node实例和TaskCancelledException。
    支持任务取消机制，在执行过程中可安全中断。

    参数:
        code: LLM生成的Python代码字符串
        node: RobotLLMNode实例（可空，会自动获取单例）
    """
    print("Inside the execute python code function")

    if node is None:
        node = RobotLLMNode.get_instance()
        if node is None:
            print("CRITICAL: Could not get node instance!")
            return

    node.get_logger().info(f"Executing generated Python code: {code}")

    try:
        # ========== 将 TaskCancelledException 传递给 exec 上下文 ==========
        exec(code, {"__builtins__": {}}, {
            "node": node,
            "TaskCancelledException": TaskCancelledException
        })
        # =================================================================
        node.get_logger().info("Code executed successfully")
    except TaskCancelledException:
        node.get_logger().warn("Code execution cancelled")
        raise  # 重新抛出，由 execute_task 处理
    except TypeError as e:
        pass
    except Exception as e:
        node.get_logger().error("Failed to execute generated code: %s", e)


def main(args=None):
    """主入口函数，初始化ROS2节点和执行器并开始旋转"""
    rclpy.init(args=args)
    node = RobotLLMNode()

    # 使用多线程执行器（4个线程）以支持并发回调处理
    executor = rclpy.executors.MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    try:
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()