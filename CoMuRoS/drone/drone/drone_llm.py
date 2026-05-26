#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
无人机LLM节点模块

该模块实现了无人机（drone）的核心控制节点，通过大语言模型(LLM)进行
任务决策和代码生成。无人机采用四旋翼（Quadrotor UAV）飞行方式，负责
在餐厅环境中执行空中监视和场景描述任务。

主要功能:
  - 订阅聊天话题（/chat/output, /chat/task_status）
  - 发布任务状态更新（/chat/task_status 和机器人特定话题）
  - 监听机器人状态消息（/robot_states，期望JSON格式）
  - 通过OpenAI GPT-4o生成任务执行代码
  - 支持悬停移动到指定位置（Hover）和场景描述（DescribeScene）
  - 集成视觉语言模型（VLM）进行图像分析和问答
  - 支持任务取消机制
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

from robot_interface.srv import GotoPoseDrone
from openai import OpenAI

# ==================== 无人机配置常量 ====================
ROBOT_NAME   = 'drone'            # 机器人名称
ROBOT_TYPE   = "Quadrotor UAV"    # 机器人类型：四旋翼无人机
NODE_NAME    = "drone_llm_node"   # ROS2节点名称
PACKAGE_NAME = "drone"            # ROS2包名称

# OpenAI API密钥（从环境变量获取）
api_key = os.getenv("DEEPSEEK_API_KEY")


class TaskCancelledException(Exception):
    """自定义异常：用于信号通知任务被取消"""
    pass


# ==================== 定义可用动作 ====================
@dataclass
class TestOption:
    """测试选项数据结构，定义无人机可执行的一个动作

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
        name="Hover",
        id=0,
        description="用于让无人机悬停或移动到指定位置。",
        example_code="node.hover(x=1.0, y=2.0, z=3.0, yaw_deg=0.0)"
    ),
    TestOption(
        name="DescribeScene",
        id=1,
        description="用于描述无人机当前视角下的场景。所有关于场景的问题必须在一个统一函数中处理，避免重复。",
        example_code="node.describe_scene(prompt='in which table is child is sitting?')"
    ),
]

class RobotLLMNode(Node):
    """
    无人机LLM核心节点（单例模式）

    该节点作为无人机的"大脑"，通过LLM进行任务决策：
      - 监听聊天相关话题（/chat/output, /chat/task_status）
      - 发布任务状态更新（/chat/task_status）和机器人特定状态（<robot_name>_task_status）
      - 镜像/消费 /robot_states 话题上的机器人状态消息（JSON格式字符串）
      - 支持通过OpenAI API动态生成和执行Python控制代码
      - 集成VLM（视觉语言模型）进行场景描述和问答
      - 订阅底部摄像头压缩图像话题进行视觉分析
      - 支持任务取消机制
    """

    _instance = None

    def __init__(self) -> None:
        """初始化无人机LLM节点，设置参数、发布器、订阅器和回调组"""
        super().__init__(NODE_NAME)

        # ---- 参数声明与获取 ----
        self.declare_parameter('robot_name', ROBOT_NAME)
        self.robot_name: str = self.get_parameter('robot_name').value
        robot_task_topic = f'{self.robot_name}_task_status'

        RobotLLMNode._instance = self

        # 初始状态变量
        self.current_time = f"Hours: {00}, Minutes: {10}, Seconds: {00}"
        self.robot_task = ""          # 当前无人机任务描述
        self.robot_states = {}        # 所有机器人的状态字典

        # VLM（视觉语言模型）客户端和图像缓存
        self.Visionclient = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        self.latest_image_b64 = None  # 最新摄像头图像的Base64编码

        # ========== 任务取消机制（标志位）==========
        self._task_cancelled = False  # 简单的布尔标志，用于信号任务取消
        # ============================================

        # ========== 回调组 ==========
        self.single_group = MutuallyExclusiveCallbackGroup()
        self.seq_group = MutuallyExclusiveCallbackGroup()
        self.multi_group = ReentrantCallbackGroup()

        # ---- 发布器（Publishers）----
        self.pub_task_status = self.create_publisher(String, '/chat/task_status', 10)
        self.pub_input_msg = self.create_publisher(String, '/chat/input', 10)
        self.pub_robot_states = self.create_publisher(String, '/robot_states', 10)
        self.pub_robot_task = self.create_publisher(String, robot_task_topic, 10)

        # GoTo服务客户端（用于调用无人机位置控制器）
        self._goto_client = self.create_client(GotoPoseDrone, '/r3/goto_pose', callback_group=self.multi_group)
        # 取消导航目标发布器
        self._cancel_goto_pub = self.create_publisher(Bool, '/r3/cancel_goto_pose_goal', 10)

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

        # 订阅无人机底部摄像头压缩图像话题（用于VLM视觉分析）
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

    def image_callback(self, msg: CompressedImage):
        """
        /r3/bottom_camera/color/image_raw/compressed 话题回调函数
        将最新的摄像头压缩图像转换为Base64编码并存储
        """
        try:
            self.latest_image_b64 = base64.b64encode(msg.data).decode("utf-8")
        except Exception as e:
            self.get_logger().error(f"Error converting image: {e}")

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
        处理任务管理器下发的任务JSON，解析并执行当前无人机的任务
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
        - self.robot_states 可能为None或各种格式的字典
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
        """发布机器人任务状态更新，并写入任务历史文件"""
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
        停止所有无人机任务
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
        使用LLM决策执行指定的无人机任务

        流程:
        1. 读取聊天历史
        2. 构建系统提示词（包含可用动作、无人机状态等信息）
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
                "   Tables 1, 2, 3, and 4 are present in the restaurant environment. "
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

    def goto_service(self, x: float, y: float, z: float, yaw_deg: float) -> bool:
        """
        调用无人机位置控制器的GoTo服务并等待结果

        非阻塞方式调用，在等待循环中支持任务取消检测。
        支持三维空间移动（x, y, z）+ 偏航角控制。

        参数:
            x: 目标X坐标
            y: 目标Y坐标
            z: 目标Z坐标（高度）
            yaw_deg: 目标偏航角（度）

        返回:
            bool: 是否成功到达目标点
        """
        # ========== 检查取消 ==========
        self.check_cancelled()
        # ========================================

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
            # ========== 循环中检查取消 ==========
            self.check_cancelled()
            # ================================================
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
        通过VLM（视觉语言模型）查询当前场景

        将底部摄像头图像发送给GPT-4.1 Vision模型并获取回答。
        回答会被发布到/chat/input话题供系统其他部分使用。

        参数:
            prompt: 用户关于场景的提问

        返回:
            bool: VLM查询是否成功
        """
        # ========== 检查取消 ==========
        self.check_cancelled()
        # ========================================

        if self.latest_image_b64 is None:
            self.get_logger().warn("No image received yet. Cannot query VLM.")
            return False

        self.get_logger().info("Sending image to VLM...")

        # 设置VLM系统提示词，指导模型识别餐桌和摊位编号
        self.system_prompt = (
            "The table numbers are from left to right 1 to 4. "
            "The stall numbers are from left to right 1 to 3. "
            "You are a vision-based event detection assistant for a Drone. "
            "Your job is to analyze the image from the drone's bottom camera and answer the user's questions based on the visual content. IN SHORT, CONCISE ANSWERS ONLY. "
        )

        try:
            # 调用GPT-4.1 Vision模型进行图像分析
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

            self.get_logger().info(f"VLM Answer: {answer}")
            # 将VLM回答发布到聊天输入话题
            answer = f"Drone (msg) | {answer}"
            self.pub_input_msg.publish(String(data=answer))
            return True

        except Exception as e:
            self.get_logger().error(f"OpenAI VLM request failed: {e}")
            return False

    # ==================== LLM可调用的动作函数 ====================

    def describe_screen(self, prompt: str = "What is in front of the drone?"):
        """
        描述无人机视角下的场景

        使用VLM分析底部摄像头图像并回答用户关于场景的问题。
        支持状态更新发布和任务取消检测。

        参数:
            prompt: 用户关于场景的提问（默认为"无人机前方是什么？"）
        """
        self.get_logger().info(f'Describing screen with prompt: "{prompt}"...')

        # 更新状态：任务开始
        self.update_robot_state({
            "current_task": "describe_screen",
            "task_status": "in_progress",
            "vision_query": prompt,
            "query_stage": "started"
        })

        # 检查是否有可用的图像
        if self.latest_image_b64 is None:
            self.get_logger().warn("No image received yet. Cannot query VLM.")
            self.update_robot_state({
                "task_status": "failed",
                "failure_reason": "no_image_available",
                "query_stage": "failed"
            })
            self.robot_task_interrupted(f"describe screen with prompt: {prompt}")
            return

        # 更新状态：正在处理图像
        self.update_robot_state({
            "task_status": "querying_vlm",
            "query_stage": "processing_image",
            "image_available": True
        })

        success = self.query_callback(prompt)

        if success:
            self.get_logger().info(f"Screen description completed.")

            # 更新状态：任务完成
            self.update_robot_state({
                "task_status": "completed",
                "query_stage": "completed",
                "last_query": prompt,
                "query_successful": True,
                "completion_timestamp": time.time()
            })

            self.robot_task_completed(f"describe screen with prompt: {prompt}")
        else:
            self.get_logger().error("Failed to get VLM response.")

            # 更新状态：任务失败
            self.update_robot_state({
                "task_status": "failed",
                "query_stage": "vlm_error",
                "failure_reason": "vlm_query_failed",
                "query_successful": False
            })

            self.robot_task_interrupted(f"describe screen with prompt: {prompt}")
        return

    def hover(self, x: float = 0.0, y: float = 0.0, z: float = 2.0, yaw_deg: float = 0.0):
        """
        控制无人机悬停或移动到指定位置

        流程:
        1. 更新状态为"正在移动"
        2. 调用goto_service导航到目标位置
        3. 根据结果更新完成或失败状态

        参数:
            x: 目标X坐标（默认0.0）
            y: 目标Y坐标（默认0.0）
            z: 目标高度（默认2.0）
            yaw_deg: 目标偏航角（度，默认0.0）
        """
        self.get_logger().info(f"Hovering at position x={x}, y={y}, z={z}, yaw={yaw_deg}...")

        # 更新状态：任务开始
        self.update_robot_state({
            "current_task": "hover",
            "task_status": "in_progress",
            "hover_stage": "started",
            "target_position": {"x": x, "y": y, "z": z, "yaw": yaw_deg}
        })

        # 更新状态：正在移动到目标位置
        self.update_robot_state({
            "task_status": "moving_to_position",
            "hover_stage": "navigating",
            "target_coords": {"x": x, "y": y, "z": z, "yaw_deg": yaw_deg}
        })

        success = self.goto_service(x=x, y=y, z=z, yaw_deg=yaw_deg)

        if success:
            self.get_logger().info(f"Hovering at position x={x}, y={y}, z={z} completed.")

            # 更新状态：悬停成功
            self.update_robot_state({
                "task_status": "completed",
                "hover_stage": "hovering",
                "current_position": {"x": x, "y": y, "z": z, "yaw": yaw_deg},
                "hover_successful": True,
                "hovering_at": f"({x}, {y}, {z})",
                "completion_timestamp": time.time()
            })

            self.robot_task_completed(f"hover at x={x}, y={y}, z={z}")
        else:
            self.get_logger().error(f"Failed to reach hover position.")

            # 更新状态：悬停失败
            self.update_robot_state({
                "task_status": "failed",
                "hover_stage": "navigation_failed",
                "failure_reason": "could_not_reach_target_position",
                "hover_successful": False,
                "attempted_position": {"x": x, "y": y, "z": z, "yaw": yaw_deg}
            })

            self.robot_task_interrupted(f"hover at x={x}, y={y}, z={z}")
        return


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