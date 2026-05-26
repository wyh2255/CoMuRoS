#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
配送机器人LLM节点模块

该模块实现了配送机器人（delivery_bot）的核心控制节点，通过大语言模型(LLM)进行
任务决策和代码生成。机器人采用差速驱动（Differential Drive）方式，负责在餐厅
环境中执行送餐和清理餐桌任务。

主要功能:
  - 订阅聊天话题（/chat/output, /chat/history, /chat/task_status）
  - 发布任务状态更新（/chat/task_status 和机器人特定话题）
  - 监听机器人状态消息（/robot_states，期望JSON格式）
  - 通过OpenAI GPT-4o生成任务执行代码
  - 支持送餐（从摊位到餐桌）和清理餐桌（从餐桌到水槽）两大动作
  - 支持任务取消机制
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
ROBOT_NAME   = 'delivery_bot'            # 机器人名称
ROBOT_TYPE   = "Differential Drive Robot" # 机器人类型：差速驱动
NODE_NAME    = "delivery_bot_llm_node"    # ROS2节点名称
PACKAGE_NAME = "delivery_bot"             # ROS2包名称


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
        name="DeliverFood",
        id=0,
        description="用于将食物从摊位送到餐桌。",
        example_code="node.deliver_food(stall_number=1, table_number=3)"
    ),
    TestOption(
        name='ClearTable',
        id=1,
        description='用于清理餐桌上的餐具并丢入水槽。从历史记录中检查哪些食物被送到该餐桌，只清理这些食物。',
        example_code='node.clear_table(table_number=2, food_name="food1")'
    )
]

# ==================== 位置常量定义 ====================
# 餐桌位置 [x, y, yaw]（共4张餐桌）
TableLocation = {
    1 : [0.0, -1.3, 0.0],
    2 : [3.0, -1.3, 0.0],
    3 : [6.0, -1.3, 0.0],
    4 : [9.0, -1.3, 0.0],
}

# 摊位位置 [x, y, yaw]（共3个摊位）
StallLocation = {
    1 : [0.0, 0.0, 0.0],
    2 : [4.0, 0.0, 0.0],
    3 : [8.0, 0.0, 0.0],
}

# 机器人归位位置
home_pose = [0.0, 0.0, 0.0]

# 水槽位置（用于清理餐具）
sink_pose = [-1.85, -0.5, 0.0]

# 食物名称列表（对应3个摊位）
food = ['food1', 'food2', 'food3']

class RobotLLMNode(Node):
    """
    配送机器人LLM核心节点（单例模式）

    该节点作为配送机器人的"大脑"，通过LLM进行任务决策：
      - 监听聊天相关话题（/chat/output, /chat/history, /chat/task_status）
      - 发布任务状态更新（/chat/task_status）和机器人特定状态（<robot_name>_task_status）
      - 镜像/消费 /robot_states 话题上的机器人状态消息（JSON格式字符串）
      - 支持通过OpenAI API动态生成和执行Python控制代码
      - 实现送餐（DeliverFood）和清理餐桌（ClearTable）两大核心功能
      - 支持任务取消机制
    """

    _instance = None

    def __init__(self) -> None:
        """初始化配送机器人LLM节点，设置参数、发布器、订阅器和回调组"""
        super().__init__(NODE_NAME)

        # ---- 参数声明与获取 ----
        self.declare_parameter('robot_name', ROBOT_NAME)
        self.robot_name: str = self.get_parameter('robot_name').value
        robot_task_topic = f'{self.robot_name}_task_status'

        RobotLLMNode._instance = self

        # 初始状态变量
        self.current_time = f"Hours: {00}, Minutes: {10}, Seconds: {00}"
        self.robot_task = ""          # 当前机器人任务描述
        self.robot_states = {}        # 所有机器人的状态字典

        # ========== 任务取消机制（标志位）==========
        self._task_cancelled = False  # 简单的布尔标志，用于信号任务取消
        # ============================================

        # ========== 回调组 ==========
        self.single_group = MutuallyExclusiveCallbackGroup()
        self.seq_group = MutuallyExclusiveCallbackGroup()
        self.multi_group = ReentrantCallbackGroup()

        # ---- 发布器（Publishers）----
        self.pub_task_status = self.create_publisher(String, '/chat/task_status', 10)
        self.pub_robot_states = self.create_publisher(String, '/robot_states', 10)
        self.pub_robot_task = self.create_publisher(String, robot_task_topic, 10)

        # GoTo服务客户端（用于调用差速驱动控制器的导航服务）
        self._goto_client = self.create_client(GotoPoseHolonomic, '/r2/goto_pose', callback_group=self.multi_group)
        # 取消导航目标发布器
        self._cancel_goto_pub = self.create_publisher(Bool, '/r2/cancel_goto_pose_goal', 10)

        # ---- 订阅器（Subscriptions）----
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

        # self.sub_chat_history = self.create_subscription(
        #     String, '/chat/history', self.on_chat_history, 10
        # )
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

        script_name = self.robot_name+"_chat_history.txt"
        self.history_file = os.path.join(package_path, directry, script_name)

        script_name = self.robot_name+'_task_history.txt'
        self.robot_task_history = os.path.join(package_path, directry, script_name)

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
                self.robot_states = rs
                self.get_logger().debug(f"robot_states keys: {list(self.robot_states.keys())}")
            else:
                self.get_logger().warning("/robot_states missing or not a dict; ignoring payload.")
        except json.JSONDecodeError:
            self.get_logger().error(f"/robot_states not JSON: {msg.data}")

    def on_tasks_json(self, msg: String) -> None:
        """
        /task_manager/tasks_json 话题回调函数
        处理任务管理器下发的任务JSON，解析并执行当前机器人的任务
        """
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

            # ========== 开始新任务前重置取消标志 ==========
            self.reset_cancellation()
            # =========================================================

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
        if robot_name is None:
            robot_name = self.robot_name

        if self.robot_states is None:
            self.robot_states = {}
            self.get_logger().info("Created top-level robot_states container dict")

        if "robot_states" not in self.robot_states:
            if all(isinstance(v, dict) for v in self.robot_states.values()) and len(self.robot_states) > 0:
                self.robot_states = {"robot_states": self.robot_states}
                self.get_logger().debug("Wrapped existing dict under 'robot_states'")
            else:
                self.robot_states["robot_states"] = {}
                self.get_logger().info("Created 'robot_states' key in top-level dict")

        robots_dict = self.robot_states["robot_states"]

        if robot_name not in robots_dict or not isinstance(robots_dict[robot_name], dict):
            robots_dict[robot_name] = {}
            self.get_logger().info(f"Created new robot entry '{robot_name}'")

        saved_state = robots_dict[robot_name]

        for key, new_val in incoming.items():
            if key not in saved_state:
                saved_state[key] = new_val
                self.get_logger().debug(f"{robot_name}: Added '{key}' = '{new_val}'")
            else:
                old_val = saved_state[key]
                if old_val != new_val:
                    saved_state[key] = new_val
                    self.get_logger().debug(f"{robot_name}: Updated '{key}' from '{old_val}' → '{new_val}'")

        msg = String()
        msg.data = json.dumps(self.robot_states)
        self.pub_robot_states.publish(msg)

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
        2. 构建系统提示词（包含可用动作、机器人状态、食物名称等信息）
        3. 调用OpenAI生成执行代码
        4. 执行生成的代码
        5. 更新任务完成状态
        """
        chat_history = self.read_chat_history()

        try:
            self.get_logger().info("Building system action messages")
            available_actions = "\n".join(
                [f"Function Name: {opt.name} \nFunction Description: {opt.description} (e.g., {opt.example_code})"
                 for opt in option_list]
            )

            self.get_logger().debug(f"Action message: {available_actions}")
            self.get_logger().info("Building system messages")

            prompt = (
                f"You are a robot control system controlling a {ROBOT_TYPE} named '{self.robot_name}'. "
                "You must generate python code to perform the task. "
                "Based on the given task, generate code using available actions."
                f"Recent Tasks (History): {chat_history} "
                f"Current States of All Robots: {self.robot_states} "
                f"Available Actions: {available_actions} "
                "Using the class reference name same as the example is important. "
                "Use the name 'node' to refer to the RobotLLMNode instance. "
                "Task Specific Rules: "
                "   'food1' is the name of the food from stall 1 "
                "   'food2' is the name of the food from stall 2 "
                "   'food3' is the name of the food from stall 3 "
                "   Table 1, 2, 3, and 4 are present in the restaurant environment. "
                "   remember this name to pick the food from stall or table "
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

    def goto_service(self, x: float, y: float, yaw_deg: float) -> bool:
        """
        调用差速驱动控制器的GoTo服务并等待结果
        非阻塞方式调用（不嵌套spin），在等待循环中支持任务取消检测
        """
        # ========== 检查取消 ==========
        self.check_cancelled()
        # ========================================

        self.get_logger().info(f"Sending goto goal: x={x}, y={y}, yaw={yaw_deg}°")

        if not self._goto_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("GotoPoseDiffDrive service NOT available!")
            return False

        req = GotoPoseHolonomic.Request()
        req.x = x
        req.y = y
        req.yaw_deg = yaw_deg

        future = self._goto_client.call_async(req)
        self.get_logger().info("Waiting for goto service response...")

        deadline = time.time() + 900000.0
        while rclpy.ok() and not future.done():
            # ========== 循环中检查取消 ==========
            self.check_cancelled()
            # ================================================
            time.sleep(0.05)
            if time.time() > deadline:
                self.get_logger().error("GotoPoseDiffDrive service call TIMED OUT.")
                return False

        try:
            res = future.result()
        except Exception as e:
            self.get_logger().error(f"Service call failed: {e}")
            return False

        if res.accepted:
            self.get_logger().info(f"Goto accepted: {res.message}")
            return True
        else:
            self.get_logger().warn(f"Goto rejected: {res.message}")
            return False

    def teleport(self, name, x, y, z):
        """
        在Ignition Gazebo仿真中传送物体位置

        调用 /world/food_court/set_pose 服务将指定名称的物体移动到目标坐标
        """
        # ========== 检查取消 ==========
        self.check_cancelled()
        # ========================================

        self.get_logger().info(f"Teleport Object: {name}")

        req = (
            f'name: "{name}", '
            f'position: {{x: {x}, y: {y}, z: {z}}}, '
            'orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}'
        )

        cmd = [
            'ign', 'service', '-s', '/world/food_court/set_pose',
            '--reqtype', 'ignition.msgs.Pose',
            '--reptype', 'ignition.msgs.Boolean',
            '--timeout', '1000',
            '--req', req
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        self.get_logger().info(f"Teleport result: {result.stdout}")
        return result.returncode == 0

    # ==================== LLM可调用的动作函数 ====================

    def deliver_food(self, stall_number, table_number):
        """
        执行送餐任务

        流程:
        1. 导航到指定摊位 -> 2. 取食物（传送） -> 3. 导航到指定餐桌
        4. 放置食物（传送） -> 5. 返回原位

        参数:
            stall_number: 摊位编号（1-3）
            table_number: 餐桌编号（1-4）
        """
        self.get_logger().info('Delivering food ...')

        # 更新状态：任务开始
        self.update_robot_state({
            "current_task": "deliver_food",
            "task_status": "in_progress",
            "stall_number": stall_number,
            "table_number": table_number,
            "delivery_stage": "started"
        })

        table_pose = TableLocation.get(table_number)
        stall_pose = StallLocation.get(stall_number)

        if table_pose is None:
            self.get_logger().error(f"Invalid table number: {table_number}")
            self.update_robot_state({
                "task_status": "failed",
                "failure_reason": f"invalid_table_number_{table_number}"
            })
            return

        if stall_pose is None:
            self.get_logger().error(f"Invalid stall number: {stall_number}")
            self.update_robot_state({
                "task_status": "failed",
                "failure_reason": f"invalid_stall_number_{stall_number}"
            })
            return

        sx, sy, syaw = stall_pose
        hx, hy, hyaw = table_pose
        home_pose_x, home_pose_y, home_pose_yaw = home_pose
        food_name = food[stall_number - 1]

        # ========== 检查取消 ==========
        self.check_cancelled()
        # ========================================

        # 更新状态：正在导航到摊位
        self.update_robot_state({
            "task_status": "navigating_to_stall",
            "delivery_stage": "going_to_stall",
            "target_coords": {"x": sx, "y": sy+0.5, "yaw": syaw},
            "food_item": food_name
        })

        result = self.goto_service(sx, sy+0.5, syaw)
        self.get_logger().info('Robot Near Stall')
        time.sleep(2.0)

        # 更新状态：正在取食物
        self.update_robot_state({
            "task_status": "picking_food",
            "delivery_stage": "at_stall",
            "current_location": f"stall_{stall_number}"
        })

        result = self.teleport(name=food_name, x=sx, y=sy+0.5, z=0.4)
        self.get_logger().info('Food Picked from Stall')
        time.sleep(3.0)

        # 更新状态：食物已取，导航到餐桌
        self.update_robot_state({
            "task_status": "navigating_to_table",
            "delivery_stage": "carrying_food",
            "food_picked": True,
            "target_coords": {"x": hx, "y": hy, "yaw": hyaw}
        })

        result = self.goto_service(hx, hy, hyaw)
        self.get_logger().info('Robot Near Table')
        time.sleep(2.0)

        # 更新状态：正在送达食物到餐桌
        self.update_robot_state({
            "task_status": "delivering_food",
            "delivery_stage": "at_table",
            "current_location": f"table_{table_number}"
        })

        result = self.teleport(name=food_name, x=hx+0.1, y=hy-0.7, z=0.6)
        self.get_logger().info("Food Delivered to table")
        time.sleep(3.0)

        # 更新状态：正在返回原位
        self.update_robot_state({
            "task_status": "returning_home",
            "delivery_stage": "going_home",
            "food_delivered": True,
            "delivery_completed_at": time.time()
        })

        result = self.goto_service(home_pose_x, home_pose_y, home_pose_yaw)
        self.get_logger().info('Robot Went Home')

        # 更新状态：任务完成
        self.update_robot_state({
            "task_status": "completed",
            "delivery_stage": "at_home",
            "current_location": "home",
            f"delivery_stall{stall_number}_to_table{table_number}": "completed",
            "completion_timestamp": time.time()
        })

    def clear_table(self, table_number, food_name):
        """
        执行清理餐桌任务

        流程:
        1. 导航到指定餐桌 -> 2. 取食物（传送） -> 3. 导航到水槽
        4. 丢入水槽（传送） -> 5. 返回原位

        参数:
            table_number: 餐桌编号（1-4）
            food_name: 要清理的食物名称
        """
        self.get_logger().info('Clearing Table ...')

        # 更新状态：任务开始
        self.update_robot_state({
            "current_task": "clear_table",
            "task_status": "in_progress",
            "table_number": table_number,
            "food_to_clear": food_name,
            "clearing_stage": "started"
        })

        table_pose = TableLocation.get(table_number)
        self.get_logger().info(f'Clear the food item: {food_name} from table number: {table_number}')

        if table_pose is None:
            self.get_logger().error(f"Invalid table number: {table_number}")
            self.update_robot_state({
                "task_status": "failed",
                "failure_reason": f"invalid_table_number_{table_number}"
            })
            return

        table_x, table_y, table_yaw = table_pose
        sink_x, sink_y, sink_yaw = sink_pose
        home_pose_x, home_pose_y, home_pose_yaw = home_pose

        # ========== 检查取消 ==========
        self.check_cancelled()
        # ========================================

        # 更新状态：导航到餐桌
        self.update_robot_state({
            "task_status": "navigating_to_table",
            "clearing_stage": "going_to_table",
            "target_coords": {"x": table_x, "y": table_y, "yaw": table_yaw}
        })

        result = self.goto_service(table_x, table_y, table_yaw)
        self.get_logger().info('Robot Near Table')
        time.sleep(2.0)

        # 更新状态：从餐桌取食物
        self.update_robot_state({
            "task_status": "picking_food_from_table",
            "clearing_stage": "at_table",
            "current_location": f"table_{table_number}"
        })

        result = self.teleport(name=food_name, x=table_x, y=table_y, z=0.4)
        self.get_logger().info('Food Picked from Table')
        time.sleep(3.0)

        # 更新状态：导航到水槽
        self.update_robot_state({
            "task_status": "navigating_to_sink",
            "clearing_stage": "carrying_dishes",
            "food_picked": True,
            "target_coords": {"x": sink_x, "y": sink_y, "yaw": sink_yaw}
        })

        result = self.goto_service(sink_x, sink_y, sink_yaw)
        self.get_logger().info('Robot Near Sink')
        time.sleep(2.0)

        # 更新状态：丢入水槽
        self.update_robot_state({
            "task_status": "dropping_in_sink",
            "clearing_stage": "at_sink",
            "current_location": "sink"
        })

        result = self.teleport(name=food_name, x=-2.5, y=-0.5, z=0.6)
        self.get_logger().info('Food Dropped in Sink')
        time.sleep(3.0)

        # 更新状态：返回原位
        self.update_robot_state({
            "task_status": "returning_home",
            "clearing_stage": "going_home",
            "dishes_cleared": True,
            "clearing_completed_at": time.time()
        })

        result = self.goto_service(home_pose_x, home_pose_y, home_pose_yaw)
        self.get_logger().info('Robot Went Home')

        # 更新状态：任务完成
        self.update_robot_state({
            "task_status": "completed",
            "clearing_stage": "at_home",
            "current_location": "home",
            f"table_{table_number}_cleared": True,
            f"cleared_item": food_name,
            "completion_timestamp": time.time()
        })


def execute_python_code(code: str, node=None):
    """
    安全执行由LLM生成的Python代码

    在受限的exec上下文中执行代码，仅暴露node实例和TaskCancelledException。
    支持任务取消机制，在执行过程中可安全中断。
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

    executor = rclpy.executors.MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    try:
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()