#!/usr/bin/env python3
"""Base ROS 2 Node with embedded A2A Worker + Mini-Agent + WebSocket registration."""

import json
import os
import sys
import threading
import time
import asyncio

from rclpy.node import Node
from std_msgs.msg import String, Bool
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from ament_index_python.packages import get_package_share_directory


class TaskCancelledException(Exception):
    pass


def _resolve_a2a_paths():
    """Resolve my_a2a and Mini-Agent paths from env vars or fallback to relative paths."""
    a2a_path = os.environ.get("MY_A2A_PATH")
    mini_agent_path = os.environ.get("MINI_AGENT_PATH")

    if not a2a_path:
        # Fallback: relative to this file's location
        a2a_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "my_a2a", "src")
    if not mini_agent_path:
        mini_agent_path = os.path.join(a2a_path, "Mini-Agent")

    return os.path.abspath(a2a_path), os.path.abspath(mini_agent_path)


_A2A_PATH, _MINI_AGENT_PATH = _resolve_a2a_paths()
if _A2A_PATH not in sys.path:
    sys.path.insert(0, _A2A_PATH)
if _MINI_AGENT_PATH not in sys.path:
    sys.path.insert(0, _MINI_AGENT_PATH)


class A2AWorkerNode(Node):
    """Base ROS Node that runs an embedded A2A HTTP server with Mini-Agent.

    On startup, registers with the Coordinator via WebSocket so the Coordinator
    can discover this Worker's capabilities via AgentCard.

    Subclasses must define:
      - _get_tools() -> list
      - _get_capabilities() -> list[str]
      - _get_system_prompt() -> str
    """

    def __init__(self, node_name: str, robot_name: str, package_name: str,
                 port: int, coordinator_url: str = "ws://localhost:8080"):
        super().__init__(node_name)

        self.robot_name = robot_name
        self._package_name = package_name
        self._a2a_port = port
        self._coordinator_url = coordinator_url
        self.current_time = "Hours: 00, Minutes: 10, Seconds: 00"
        self.robot_task = ""
        self.robot_states = {}

        self._task_cancelled = False
        self._a2a_server = None
        self._a2a_thread = None
        self._ws_thread = None

        # Callback groups
        self.single_group = MutuallyExclusiveCallbackGroup()
        self.seq_group = MutuallyExclusiveCallbackGroup()
        self.multi_group = ReentrantCallbackGroup()

        robot_task_topic = f"{self.robot_name}_task_status"

        # Publishers
        self.pub_task_status = self.create_publisher(String, "/chat/task_status", 10)
        self.pub_robot_states = self.create_publisher(String, "/robot_states", 10)
        self.pub_robot_task = self.create_publisher(String, robot_task_topic, 10)

        # Subscribers
        self.sub_robot_states = self.create_subscription(
            String, "/robot_states", self.on_robot_states, 10, callback_group=self.seq_group)
        self.current_time_sub = self.create_subscription(
            String, "/current_time", self.on_current_time, 10, callback_group=self.seq_group)
        self.sub_tasks_json = self.create_subscription(
            String, "/task_manager/tasks_json", self.on_tasks_json, 10, callback_group=self.multi_group)
        self.sub_chat_output = self.create_subscription(
            String, "/chat/output", self.on_chat_output, 10, callback_group=self.single_group)
        self.sub_chat_task_status = self.create_subscription(
            String, "/chat/task_status", self.on_chat_task_status, 10)

        # File paths
        directry = "data"
        package_path = get_package_share_directory(package_name)
        self.history_file = os.path.join(package_path, directry, f"{robot_name}_chat_history.txt")
        self.robot_task_history = os.path.join(package_path, directry, f"{robot_name}_task_history.txt")

        self.clear_files()
        self.robot_has_no_current_task()

    # --- Subclass hooks ---
    def _get_tools(self) -> list:
        raise NotImplementedError

    def _get_system_prompt(self) -> str:
        return ""

    def _get_capabilities(self) -> list[str]:
        return []

    def _get_model(self) -> str:
        return "deepseek-v4-flash"

    # --- A2A Server + WebSocket registration ---
    def start_a2a_server(self):
        """Start A2A HTTP server and register with Coordinator via WebSocket."""
        from openharness_a2a.worker.a2a_server import create_worker_a2a_server

        extra_tools = self._get_tools()
        system_prompt = self._get_system_prompt_with_history()  # includes chat history
        host = "0.0.0.0"

        server = create_worker_a2a_server(
            worker_id=self.robot_name,
            host=host,
            port=self._a2a_port,
            capabilities=self._get_capabilities(),
            backend="mini_agent",
            model=self._get_model(),
            extra_tools=extra_tools,
            system_prompt=system_prompt,
            step_callback=self._make_step_callback(),  # publishes to /chat/output
        )
        self._a2a_server = server

        # Start uvicorn in daemon thread
        async def _serve():
            await server.serve()

        def _run_uvicorn():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(_serve())

        self._a2a_thread = threading.Thread(target=_run_uvicorn, daemon=True)
        self._a2a_thread.start()
        self.get_logger().info(f"A2A HTTP server started on port {self._a2a_port}")

        # Start WebSocket registration in daemon thread
        a2a_endpoint = f"http://{host}:{self._a2a_port}/"

        async def _ws_register():
            import websockets
            ws_url = f"{self._coordinator_url}/ws/worker/{self.robot_name}"
            for attempt in range(3):
                try:
                    async with websockets.connect(ws_url) as ws:
                        # Send WS_REGISTER
                        await ws.send(json.dumps({
                            "type": "register",
                            "payload": {
                                "worker_id": self.robot_name,
                                "a2a_endpoint": a2a_endpoint,
                            },
                        }))
                        self.get_logger().info(f"Registered with Coordinator at {ws_url}")
                        # Heartbeat loop
                        while True:
                            await asyncio.sleep(30)
                            await ws.send(json.dumps({
                                "type": "heartbeat",
                                "payload": {"worker_id": self.robot_name},
                            }))
                except Exception as e:
                    self.get_logger().warning(
                        f"Coordinator WS attempt {attempt+1}/3 failed: {e}"
                    )
                    if attempt < 2:
                        await asyncio.sleep(2 ** attempt)  # 1s, 2s backoff
            self.get_logger().error(
                f"Coordinator WebSocket registration failed after 3 attempts"
            )

        def _run_ws():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(_ws_register())

        self._ws_thread = threading.Thread(target=_run_ws, daemon=True)
        self._ws_thread.start()
        self.get_logger().info(f"WebSocket registration thread started for {self.robot_name}")

    # --- Chat history + agent output publishing ---
    # IMPORTANT: Chat history continuity fix.
    # Without this, the Mini-Agent has no memory of past messages because
    # A2A tasks are stateless and Worker responses never reach /chat/output.
    # We inject read_chat_history() into every system prompt AND publish
    # the Mini-Agent's step callback output back to /chat/output so the
    # history file records both user messages and robot replies.

    def _get_system_prompt_with_history(self) -> str:
        """Combine subclass system prompt with chat history for context."""
        base_prompt = self._get_system_prompt()
        history = self.read_chat_history()
        if history and history != "No previous chat history.":
            base_prompt += f"\n\nChat history so far:\n{history}"
        return base_prompt

    def _make_step_callback(self):
        """Return a callback that publishes Mini-Agent output to /chat/output.

        This ensures Worker responses are recorded in chat history files,
        keeping the conversation context alive across A2A task invocations.
        """

        # Lazily create the publisher on first use
        _pub = None

        def step_callback(agent_step):
            nonlocal _pub
            if _pub is None:
                _pub = self.create_publisher(String, "/chat/output", 10)
            msg = String()
            # Format: "agent|step_output" so on_chat_output() writes it to history
            content = getattr(agent_step, "content", str(agent_step))
            msg.data = f"{self.robot_name}|{content}"
            _pub.publish(msg)

        return step_callback

    # --- Task cancellation ---
    def check_cancelled(self):
        if self._task_cancelled:
            self.get_logger().warn("Task cancellation detected!")
            raise TaskCancelledException("Task was cancelled")

    def reset_cancellation(self):
        self._task_cancelled = False

    # --- File management ---
    def clear_files(self):
        for path in [self.history_file, self.robot_task_history]:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                f.write("")

    # --- ROS Callbacks ---
    def on_robot_states(self, msg: String):
        try:
            data = json.loads(msg.data)
            rs = data.get("robot_states")
            if isinstance(rs, dict):
                self.robot_states = rs
        except json.JSONDecodeError:
            pass

    def on_tasks_json(self, msg: String):
        """Legacy task_manager JSON tasks — no-op in A2A mode."""
        pass

    def on_chat_output(self, msg: String):
        timestamp = self.current_time
        parts = msg.data.split("|", 1)
        if len(parts) == 2:
            role, content = parts[0].strip(), parts[1].strip()
            self.chat_entry = f"[Time: {timestamp}] {role.capitalize()}: {content}"
        else:
            self.chat_entry = f"[Time: {timestamp}] Task Manager:\n{msg.data}"
        with open(self.history_file, "a") as file:
            file.write(self.chat_entry + "\n")

    def on_chat_task_status(self, msg: String):
        with open(self.history_file, "a") as file:
            file.write(msg.data + "\n")

    def on_current_time(self, msg: String):
        self.current_time = msg.data

    # --- Robot state management ---
    def update_robot_state(self, incoming: dict, robot_name: str | None = None):
        if robot_name is None:
            robot_name = self.robot_name
        if self.robot_states is None:
            self.robot_states = {}
        if "robot_states" not in self.robot_states:
            self.robot_states["robot_states"] = {}
        robots_dict = self.robot_states["robot_states"]
        if robot_name not in robots_dict or not isinstance(robots_dict[robot_name], dict):
            robots_dict[robot_name] = {}
        saved_state = robots_dict[robot_name]
        for key, new_val in incoming.items():
            if key not in saved_state:
                saved_state[key] = new_val
            elif saved_state[key] != new_val:
                saved_state[key] = new_val
        msg = String()
        msg.data = json.dumps(self.robot_states)
        self.pub_robot_states.publish(msg)

    # --- Task status management ---
    def robot_task_in_progress(self, robot_task=None):
        if not robot_task:
            robot_task = self.robot_task
        task_msg = f"{self.robot_name.capitalize()} (status) : {robot_task} : TASK IN PROGRESS"
        self._publish_task_status(task_msg)

    def robot_task_completed(self, robot_task=None):
        if not robot_task:
            robot_task = self.robot_task
        task_msg = f"{self.robot_name.capitalize()} (status) : {robot_task} : TASK COMPLETED"
        self._publish_task_status(task_msg)

    def robot_task_interrupted(self, robot_task=None):
        if not robot_task:
            robot_task = self.robot_task
        task_msg = f"{self.robot_name.capitalize()} (status) : {robot_task} : TASK INTERRUPTED"
        self._publish_task_status(task_msg)

    def robot_has_no_current_task(self):
        task_msg = f"{self.robot_name.capitalize()} (status) :  : NO CURRENT TASK"
        self._publish_task_status(task_msg)

    def tasks_completed(self, task=None):
        msg = String()
        msg.data = f"{self.robot_name.capitalize()} (status): ALL TASKS COMPLETED"
        self.pub_task_status.publish(msg)

    def stop_tasks(self):
        self._task_cancelled = True

    def _publish_task_status(self, status_msg: str):
        msg = String()
        msg.data = status_msg
        self.pub_robot_task.publish(msg)
        try:
            with open(self.robot_task_history, "a") as file:
                file.write(f"{status_msg}\n")
        except FileNotFoundError:
            pass

    def read_chat_history(self) -> str:
        if not os.path.exists(self.history_file):
            return "No previous chat history."
        with open(self.history_file, "r", encoding="utf-8") as file:
            history = file.read().strip()
        return history or "No previous chat history."

    def destroy_node(self):
        self.get_logger().info("Shutting down A2A worker...")
        super().destroy_node()
