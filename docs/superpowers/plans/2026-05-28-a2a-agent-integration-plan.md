# A2A Agent Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace all hard-coded LLM API calls in CoMuRoS with A2A Agent-based architecture: Coordinator replaces task_manager.py, each robot node becomes an A2A Worker with Mini-Agent + Tool Calling.

**Architecture:** Coordinator (independent process, FastAPI :8080) performs LLM-driven task decomposition via RouterAgent. Four robot Workers (ROS nodes with embedded A2A HTTP server + WebSocket registration + Mini-Agent) execute tasks via Tool Calling → ROS services. GUI sends user input to Coordinator via HTTP POST `/tasks`, polls `/tasks/{task_id}` for results.

**Tech Stack:** ROS 2 (Python), FastAPI + uvicorn, A2A SDK (>1.0.0), Mini-Agent (MiniMax), DeepSeek API, websockets, httpx

**New dependencies to install:**
```bash
pip install httpx websockets
```
Add `httpx` to `CoMuRoS/requirements.txt` and `CoMuRoS/chatty/setup.py` install_requires. (`websockets` is already a my_a2a dependency.)

---

### Task 1: Enhance MiniAgentAdapter to accept custom tools and system prompt

**Files:**
- Modify: `my_a2a/src/openharness_a2a/worker/mini_agent_adapter.py`
- Modify: `my_a2a/src/openharness_a2a/worker/a2a_server.py`

- [ ] **Step 1: Add `extra_tools`, `system_prompt`, and `step_callback` parameters to MiniAgentAdapter**

Edit `my_a2a/src/openharness_a2a/worker/mini_agent_adapter.py`:

```python
# Update __init__ signature — add extra_tools and step_callback parameters:
def __init__(
    self,
    model: str = "MiniMax-M2.7",
    system_prompt: str = "",
    max_steps: int = 50,
    workspace_dir: str = "./workspace",
    extra_tools: list | None = None,
    step_callback: callable | None = None,
):
    self._model = model
    self._system_prompt = system_prompt or ""
    self._max_steps = max_steps
    self._workspace_dir = Path(workspace_dir)
    self._agent: Agent | None = None
    self._extra_tools = extra_tools or []
    self._step_callback = step_callback
```

In `_build_agent()`, after the standard tools list, append extra_tools (in both the `config is not None` and `config is None` branches):

```python
# In the config-path branch:
            tools = [
                ReadTool(),
                WriteTool(),
                BashTool(workspace_dir=config.agent.workspace_dir),
            ]
            if self._extra_tools:
                tools.extend(self._extra_tools)

# In the config-is-None fallback branch:
            tools = [ReadTool(), WriteTool()]
            if self._extra_tools:
                tools.extend(self._extra_tools)
```

In both branches, pass the step_callback to the Agent constructor if the framework supports it. Mini-Agent's Agent accepts a `step_callback` keyword arg that is called after each step with the agent step result:

```python
# In both branches, if self._step_callback:
            agent_kwargs = {}
            if self._step_callback:
                agent_kwargs["step_callback"] = self._step_callback
            agent = Agent(
                llm_client=llm_client,
                tools=tools,
                system_prompt=system_prompt,
                max_steps=self._max_steps if self._max_steps else None,
                **agent_kwargs,
            )
```

The `system_prompt` is already used via `self._system_prompt` (already exists in the code). Verify existing usage; no change needed there.

- [ ] **Step 2: Accept extra_tools, system_prompt, and step_callback in create_worker_a2a_server**

Edit `my_a2a/src/openharness_a2a/worker/a2a_server.py`:

```python
def create_worker_a2a_server(
    worker_id: str,
    host: str = "0.0.0.0",
    port: int = 8090,
    capabilities: list[str] | None = None,
    backend: str = "openharness",
    model: str = "claude-opus-4-5",
    extra_tools: list | None = None,
    system_prompt: str = "",
    step_callback: callable | None = None,
) -> uvicorn.Server:
```

And pass them through:

```python
    if backend == "mini_agent":
        executor = MiniAgentAdapter(
            model=model,
            system_prompt=system_prompt,
            extra_tools=extra_tools,
            step_callback=step_callback,
        )
```

- [ ] **Step 3: Run existing tests to verify no regression**

```bash
cd my_a2a && python -m pytest tests/ -v --timeout=30 2>&1 | tail -30
```

Expected: existing tests still pass.

- [ ] **Step 4: Commit**

```bash
cd my_a2a && git add src/openharness_a2a/worker/mini_agent_adapter.py src/openharness_a2a/worker/a2a_server.py
git commit -m "feat(worker): add extra_tools and system_prompt parameters to MiniAgentAdapter"
```

---

### Task 2: Create base A2AWorkerNode class with WebSocket registration and chat history

**Files:**
- Create: `CoMuRoS/CoMuRoS/robot_llm/robot_llm/a2a_worker_node.py`

- [ ] **Step 1: Create the base A2AWorkerNode**

Create `CoMuRoS/CoMuRoS/robot_llm/robot_llm/a2a_worker_node.py`:

```python
#!/usr/bin/env python3
"""Base ROS 2 Node with embedded A2A Worker + Mini-Agent + WebSocket registration."""

import json
import os
import sys
import threading
import time
import asyncio

import rclpy
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
        import functools
        from std_msgs.msg import String

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
```

- [ ] **Step 2: Commit**

```bash
cd CoMuRoS && git add CoMuRoS/robot_llm/robot_llm/a2a_worker_node.py
git commit -m "feat(robot_llm): add base A2AWorkerNode with WebSocket registration and system_prompt wiring"
```

---

### Task 3: Create cleaning_bot tools and A2A Worker

**Files:**
- Create: `CoMuRoS/CoMuRoS/cleaning_bot/cleaning_bot/tools.py`
- Rewrite: `CoMuRoS/CoMuRoS/cleaning_bot/cleaning_bot/cleaning_bot_llm.py`

- [ ] **Step 1: Create cleaning bot Tool**

Create `CoMuRoS/CoMuRoS/cleaning_bot/cleaning_bot/tools.py`:

```python
"""Mini-Agent Tool definitions for cleaning_bot."""

from mini_agent.tools.base import Tool, ToolResult


class CleanTool(Tool):
    """Execute restaurant cleaning along predefined path."""

    def __init__(self, node):
        self._node = node

    @property
    def name(self) -> str:
        return "clean"

    @property
    def description(self) -> str:
        return (
            "Clean the restaurant by navigating through 6 predefined cleaning points: "
            "(11.0,-3.0)->(4.0,-3.0)->(-3.5,-3.0)->(-3.5,3.0)->(11.0,3.0)->(11.0,0.0). "
            "Removes obstacles encountered during cleaning. Returns success/failure status."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {},
            "required": [],
        }

    async def execute(self, **kwargs) -> ToolResult:
        try:
            self._node.clean()
            return ToolResult(success=True, content="Restaurant cleaning completed successfully.")
        except Exception as e:
            return ToolResult(success=False, content="", error=str(e))
```

- [ ] **Step 2: Rewrite cleaning_bot_llm.py as A2AWorkerNode subclass**

Rewrite `CoMuRoS/CoMuRoS/cleaning_bot/cleaning_bot/cleaning_bot_llm.py` — replace the entire file content. Keep only the robot-specific methods (`goto_service`, `remove_cube`, `clean`) and extend `A2AWorkerNode`. Remove all LLM code: `generate_action_prompt()`, `execute_task()`, `execute_python_code()`, `TestOption`/`option_list`, all `call_openai()` calls, all OpenAI/DeepSeek imports.

```python
#!/usr/bin/env python3
"""Cleaning bot A2A Worker node."""

import time
import subprocess

import rclpy
from rclpy.executors import MultiThreadedExecutor
from robot_interface.srv import GotoPoseHolonomic
from std_msgs.msg import Bool

from robot_llm.a2a_worker_node import A2AWorkerNode, TaskCancelledException
from cleaning_bot.tools import CleanTool


ROBOT_NAME = "cleaning_bot"
ROBOT_TYPE = "Holonomic Drive Robot"
PACKAGE_NAME = "cleaning_bot"
NODE_NAME = "cleaning_bot_llm_node"
A2A_PORT = 8090


class CleaningBotWorker(A2AWorkerNode):
    """Cleaning bot: A2A Worker with CleanTool."""

    def __init__(self):
        super().__init__(
            node_name=NODE_NAME,
            robot_name=ROBOT_NAME,
            package_name=PACKAGE_NAME,
            port=A2A_PORT,
        )

        # GoTo service client
        self._goto_client = self.create_client(
            GotoPoseHolonomic, "/r1/goto_pose", callback_group=self.multi_group
        )
        self._cancel_goto_pub = self.create_publisher(Bool, "/r1/cancel_goto_pose_goal", 10)

        self.start_a2a_server()

    def _get_tools(self) -> list:
        return [CleanTool(node=self)]

    def _get_capabilities(self) -> list[str]:
        return ["cleaning", "obstacle_removal"]

    def _get_system_prompt(self) -> str:
        return (
            f"You control a {ROBOT_TYPE} named '{ROBOT_NAME}' in a restaurant food court. "
            "Use the 'clean' tool to clean the restaurant by navigating through all predefined points. "
            "Report task status updates when completing each action."
        )

    def goto_service(self, x: float, y: float, yaw_deg: float) -> bool:
        self.check_cancelled()
        if not self._goto_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("GotoPoseHolonomic service NOT available!")
            return False
        req = GotoPoseHolonomic.Request()
        req.x = x; req.y = y; req.yaw_deg = yaw_deg
        future = self._goto_client.call_async(req)
        deadline = time.time() + 1000000.0
        while rclpy.ok() and not future.done():
            self.check_cancelled()
            time.sleep(0.05)
            if time.time() > deadline:
                return False
        try:
            res = future.result()
        except Exception:
            return False
        return res.success if res.accepted else False

    def remove_cube(self):
        cmd = [
            "ign", "service", "-s", "/world/food_court/remove",
            "--reqtype", "ignition.msgs.Entity",
            "--reptype", "ignition.msgs.Boolean",
            "--timeout", "1000",
            "--req", 'name: "small_cube", type: 2',
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.returncode == 0

    def clean(self):
        """Execute restaurant cleaning task."""
        self.update_robot_state({
            "current_task": "clean_restaurant",
            "task_status": "in_progress",
            "cleaning_progress": "0/6",
        })

        cleaning_locations = [
            (11.0, -3.0, 0.0), (4.0, -3.0, 0.0), (-3.5, -3.0, 0.0),
            (-3.5, 3.0, 0.0), (11.0, 3.0, 0.0), (11.0, 0.0, 0.0),
        ]

        for idx, (x, y, yaw) in enumerate(cleaning_locations):
            self.check_cancelled()
            self.update_robot_state({
                "cleaning_progress": f"{idx}/6",
                "target_coords": {"x": x, "y": y, "yaw": yaw},
            })
            if not self.goto_service(x=x, y=y, yaw_deg=yaw):
                self.robot_task_interrupted("clean")
                raise RuntimeError(f"Failed to reach cleaning location {idx+1}")

            if idx == 1:
                self.remove_cube()
            time.sleep(2.0)
            self.update_robot_state({
                "cleaning_progress": f"{idx+1}/6",
                f"location_{idx+1}_cleaned": True,
            })

        self.update_robot_state({"task_status": "completed", "cleaning_progress": "6/6"})
        self.robot_task_completed("clean")


def main(args=None):
    rclpy.init(args=args)
    node = CleaningBotWorker()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Commit**

```bash
cd CoMuRoS && git add CoMuRoS/cleaning_bot/cleaning_bot/tools.py CoMuRoS/cleaning_bot/cleaning_bot/cleaning_bot_llm.py
git commit -m "feat(cleaning_bot): refactor to A2A Worker with CleanTool"
```

---

### Task 4: Create delivery_bot tools and A2A Worker

**Files:**
- Create: `CoMuRoS/CoMuRoS/delivery_bot/delivery_bot/tools.py`
- Rewrite: `CoMuRoS/CoMuRoS/delivery_bot/delivery_bot/delivery_bot_llm.py`

- [ ] **Step 1: Create delivery_bot tools**

Create `CoMuRoS/CoMuRoS/delivery_bot/delivery_bot/tools.py`:

```python
"""Mini-Agent Tool definitions for delivery_bot."""

from mini_agent.tools.base import Tool, ToolResult


class DeliverFoodTool(Tool):
    """Deliver food from stall to table."""

    def __init__(self, node):
        self._node = node

    @property
    def name(self) -> str:
        return "deliver_food"

    @property
    def description(self) -> str:
        return (
            "Deliver food from a stall to a table. "
            "Navigates: home -> stall -> table -> home. "
            "Uses teleport to pick/place food objects in simulation."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "stall_number": {"type": "integer", "description": "Stall number 1-3"},
                "table_number": {"type": "integer", "description": "Table number 1-4"},
            },
            "required": ["stall_number", "table_number"],
        }

    async def execute(self, stall_number: int, table_number: int, **kwargs) -> ToolResult:
        try:
            self._node.deliver_food(stall_number, table_number)
            food_names = ["food1", "food2", "food3"]
            return ToolResult(
                success=True,
                content=f"Delivered {food_names[stall_number-1]} from stall {stall_number} to table {table_number}.",
            )
        except Exception as e:
            return ToolResult(success=False, content="", error=str(e))


class ClearTableTool(Tool):
    """Clear dishes from table to sink."""

    def __init__(self, node):
        self._node = node

    @property
    def name(self) -> str:
        return "clear_table"

    @property
    def description(self) -> str:
        return (
            "Clear dishes from a table and drop them in the sink. "
            "Navigates: home -> table -> sink -> home. "
            "Check chat history to determine which food was delivered to the table."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "table_number": {"type": "integer", "description": "Table number 1-4"},
                "food_name": {"type": "string", "description": "Name of food to clear (e.g. 'food1')"},
            },
            "required": ["table_number", "food_name"],
        }

    async def execute(self, table_number: int, food_name: str, **kwargs) -> ToolResult:
        try:
            self._node.clear_table(table_number, food_name)
            return ToolResult(
                success=True,
                content=f"Cleared {food_name} from table {table_number} to sink.",
            )
        except Exception as e:
            return ToolResult(success=False, content="", error=str(e))
```

- [ ] **Step 2: Rewrite delivery_bot_llm.py**

Rewrite `CoMuRoS/CoMuRoS/delivery_bot/delivery_bot/delivery_bot_llm.py`. Extend `A2AWorkerNode`. Robot name `delivery_bot`, port `8091`. `_get_tools()` returns `[DeliverFoodTool(node=self), ClearTableTool(node=self)]`. `_get_capabilities()` returns `["food_delivery", "table_clearing"]`. Keep `goto_service()`, `teleport()` methods. Modify `deliver_food()` and `clear_table()` to raise exceptions on failure instead of silently returning `None` (currently they return void on invalid stall/table numbers). Keep constants `TableLocation`, `StallLocation`, `home_pose`, `sink_pose`, `food`. Remove all LLM code generation, OpenAI/DeepSeek imports, `TestOption`/`option_list`, `execute_task()`, `execute_python_code()`, `generate_action_prompt()`.

For `deliver_food()`, change the invalid-input early returns (currently `return` without value at the `table_pose is None` and `stall_pose is None` checks) to `raise ValueError(...)`. For `clear_table()`, change the invalid table check similarly.

(The full file content is structurally identical to cleaning_bot — extend A2AWorkerNode, call super().__init__ with robot-specific params, implement hooks, keep robot methods, start A2A server. Omitted here for brevity; the pattern is established in Task 3.)

- [ ] **Step 3: Commit**

```bash
cd CoMuRoS && git add CoMuRoS/delivery_bot/delivery_bot/tools.py CoMuRoS/delivery_bot/delivery_bot/delivery_bot_llm.py
git commit -m "feat(delivery_bot): refactor to A2A Worker with DeliverFood and ClearTable Tools"
```

---

### Task 5: Create drone tools and A2A Worker

**Files:**
- Create: `CoMuRoS/CoMuRoS/drone/drone/tools.py`
- Rewrite: `CoMuRoS/CoMuRoS/drone/drone/drone_llm.py`

- [ ] **Step 1: Create drone tools**

Create `CoMuRoS/CoMuRoS/drone/drone/tools.py`:

```python
"""Mini-Agent Tool definitions for drone."""

from mini_agent.tools.base import Tool, ToolResult


class HoverTool(Tool):
    """Move drone to a 3D position."""

    def __init__(self, node):
        self._node = node

    @property
    def name(self) -> str:
        return "hover"

    @property
    def description(self) -> str:
        return "Move drone to specified 3D position. All coordinates in meters, yaw in degrees."

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "x": {"type": "number", "description": "Target X coordinate"},
                "y": {"type": "number", "description": "Target Y coordinate"},
                "z": {"type": "number", "description": "Target Z (altitude in meters)"},
                "yaw_deg": {"type": "number", "description": "Target yaw angle in degrees"},
            },
            "required": ["x", "y", "z", "yaw_deg"],
        }

    async def execute(self, x: float, y: float, z: float, yaw_deg: float, **kwargs) -> ToolResult:
        try:
            self._node.hover(x=x, y=y, z=z, yaw_deg=yaw_deg)
            return ToolResult(success=True, content=f"Hovered to x={x}, y={y}, z={z}.")
        except Exception as e:
            return ToolResult(success=False, content="", error=str(e))


class DescribeSceneTool(Tool):
    """Describe what the drone sees via bottom camera and VLM."""

    def __init__(self, node):
        self._node = node

    @property
    def name(self) -> str:
        return "describe_screen"

    @property
    def description(self) -> str:
        return "Analyze the drone's bottom camera image and answer a question about the scene."

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Question about the scene (e.g., 'which table has food on it?')",
                },
            },
            "required": ["prompt"],
        }

    async def execute(self, prompt: str, **kwargs) -> ToolResult:
        try:
            # Call query_callback directly to get the VLM text response
            success = self._node.query_callback(prompt)
            if success:
                # describe_screen() publishes VLM output to /chat/input;
                # the VLM response is also available as the last chat entry.
                # Read the latest from chat history to capture it.
                history = self._node.read_chat_history()
                lines = history.strip().split("\n")
                last_msg = lines[-1] if lines else ""
                return ToolResult(
                    success=True,
                    content=f"Scene analysis completed. Latest observation: {last_msg}",
                )
            else:
                return ToolResult(success=False, content="", error="VLM query failed")
        except Exception as e:
            return ToolResult(success=False, content="", error=str(e))
```

- [ ] **Step 2: Rewrite drone_llm.py**

Rewrite `CoMuRoS/CoMuRoS/drone/drone/drone_llm.py`. Extend `A2AWorkerNode`. Robot name `drone`, port `8092`. `_get_tools()` returns `[HoverTool(node=self), DescribeSceneTool(node=self)]`. `_get_capabilities()` returns `["aerial_inspection", "scene_description"]`. Keep `hover()`, `goto_service()`, `describe_screen()`, `query_callback()`, `image_callback()`, VLM client. Remove all LLM code generation, `TestOption`/`option_list`, `execute_task()`, `execute_python_code()`, `generate_action_prompt()`.

- [ ] **Step 3: Commit**

```bash
cd CoMuRoS && git add CoMuRoS/drone/drone/tools.py CoMuRoS/drone/drone/drone_llm.py
git commit -m "feat(drone): refactor to A2A Worker with Hover and DescribeScene Tools"
```

---

### Task 6: Create robot_arm tools and A2A Worker

**Files:**
- Create: `CoMuRoS/CoMuRoS/robot_llm/robot_llm/tools.py`
- Rewrite: `CoMuRoS/CoMuRoS/robot_llm/robot_llm/robot_llm.py`

- [ ] **Step 1: Create robot arm Tool**

Create `CoMuRoS/CoMuRoS/robot_llm/robot_llm/tools.py`:

```python
"""Mini-Agent Tool definitions for robot arm."""

from mini_agent.tools.base import Tool, ToolResult


class PickObjectTool(Tool):
    """Pick an object by color name."""

    def __init__(self, node):
        self._node = node

    @property
    def name(self) -> str:
        return "pick_object"

    @property
    def description(self) -> str:
        return "Pick an object by color. Valid objects: 'green object', 'brown object', 'grey object'."

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "object_name": {
                    "type": "string",
                    "description": "Object to pick: 'green object', 'brown object', or 'grey object'",
                },
            },
            "required": ["object_name"],
        }

    async def execute(self, object_name: str, **kwargs) -> ToolResult:
        try:
            success = self._node.pick_object(object_name)
            if success:
                return ToolResult(success=True, content=f"Picked {object_name} successfully.")
            else:
                return ToolResult(success=False, content="", error=f"Failed to pick {object_name}")
        except Exception as e:
            return ToolResult(success=False, content="", error=str(e))
```

- [ ] **Step 2: Rewrite robot_llm.py**

Rewrite `CoMuRoS/CoMuRoS/robot_llm/robot_llm/robot_llm.py`. Extend `A2AWorkerNode`. Robot name `robot1`, port `8093`. `_get_tools()` returns `[PickObjectTool(node=self)]`. `_get_capabilities()` returns `["pick_and_place"]`. Keep `pick_object()`, `pick_green_object()`, `pick_brown_object()`, `pick_grey_object()`, StartPick service client. Remove all LLM code generation, `TestOption`/`option_list`, `execute_task()`, `execute_python_code()`, `generate_action_prompt()`.

Note: The robot_arm (`robot1`) is not currently launched by `robot_llms.launch.py` (which launches cleaning_bot, delivery_bot, drone). Add it as a new Node entry:

```python
# In robot_llms.launch.py, add:
Node(
    package='robot_llm',
    executable='robot_llm',
    name='robot1_llm',
    output='screen',
),
```

- [ ] **Step 3: Commit**

```bash
cd CoMuRoS && git add CoMuRoS/robot_llm/robot_llm/tools.py CoMuRoS/robot_llm/robot_llm/robot_llm.py
git commit -m "feat(robot_llm): refactor robot arm to A2A Worker with PickObject Tool"
```

---

### Task 7: Configure Coordinator with robot agents and enhance RouterAgent

**Files:**
- Modify: `my_a2a/config/agents.yaml`
- Modify: `my_a2a/src/openharness_a2a/coordinator/router.py`

- [ ] **Step 1: Add robot entries to agents.yaml**

Replace `my_a2a/config/agents.yaml`:

```yaml
agents:
  - id: cleaning-bot
    description: "Holonomic drive robot that cleans the restaurant floor along predefined paths, can remove obstacles"
    endpoint: http://localhost:8090
    capabilities: [cleaning, obstacle_removal]
    backend: mini_agent
    model: deepseek-v4-flash

  - id: delivery-bot
    description: "Differential drive robot that delivers food from stalls to tables and clears tables to sink"
    endpoint: http://localhost:8091
    capabilities: [food_delivery, table_clearing]
    backend: mini_agent
    model: deepseek-v4-flash

  - id: drone
    description: "Quadrotor UAV that hovers to positions and describes scenes via bottom camera (VLM)"
    endpoint: http://localhost:8092
    capabilities: [aerial_inspection, scene_description]
    backend: mini_agent
    model: deepseek-v4-flash

  - id: robot-arm
    description: "Fixed robotic arm that picks objects by color (green, brown, grey)"
    endpoint: http://localhost:8093
    capabilities: [pick_and_place]
    backend: mini_agent
    model: deepseek-v4-flash
```

- [ ] **Step 2: Enhance RouterAgent system prompt for robot task planning + inject dynamic Worker status**

Edit `my_a2a/src/openharness_a2a/coordinator/router.py`.

First, replace `ROUTER_SYSTEM_PROMPT`:

```python
ROUTER_SYSTEM_PROMPT = """You are an intelligent task planning agent for a multi-robot team in a restaurant food court.

The environment has:
- 3 food stalls (stall 1, 2, 3 from left to right)
- 4 tables (table 1, 2, 3, 4 from left to right)
- 1 sink for dish disposal
- food items named 'food1' (stall1), 'food2' (stall2), 'food3' (stall3)

Available Robots:
{agents_text}

For each user request, output JSON format:
{{
  "subtasks": [
    {{"agent_id": "robot-id", "prompt": "Complete natural language instruction for this robot"}},
    ...
  ],
  "execution_order": "sequential",
  "reasoning": "Why this plan"
}}

Rules:
- Simple single-robot task: execution_order = "single"
- Multi-step sequential tasks: execution_order = "sequential"
- Independent parallel tasks: execution_order = "parallel"
- Each subtask prompt must be a complete, self-contained instruction
- Consider robot capabilities and physical constraints (payload, reach, speed)
- If task requires coordination (e.g., drone inspects THEN delivery bot clears), use sequential
- If event indicates task interruption, cancel current tasks and re-plan
- Only output JSON, no other text
"""
```

Then, modify `_build_system_prompt()` to inject dynamic Worker status from WorkerRegistry (which tracks online/offline state via WebSocket heartbeat):

```python
# In RouterAgent class, enhance _build_system_prompt():
def _build_system_prompt(self, user_request: str | None = None) -> str:
    agents_text = self._registry.get_all_agents_prompt_text()
    
    # === NEW: Inject dynamic Worker status from WorkerRegistry ===
    # WorkerRegistry tracks online state via WebSocket heartbeat (30s interval).
    # Inject current status so RouterAgent doesn't assign tasks to offline Workers.
    worker_status_lines = []
    try:
        registry = self._registry  # AgentRegistry
        if hasattr(registry, '_worker_registry') and registry._worker_registry:
            wr = registry._worker_registry
            for wid, info in wr.items():
                status = "online" if info.get("online") else "offline"
                busy = "(busy)" if info.get("busy") else "(idle)"
                worker_status_lines.append(f"  - {wid}: {status} {busy}")
    except Exception:
        pass  # safe fallback: no status injection if registry not available
    
    status_text = ""
    if worker_status_lines:
        status_text = "\nCurrent Worker status:\n" + "\n".join(worker_status_lines)
    # ============================================================
    
    return ROUTER_SYSTEM_PROMPT.format(agents_text=agents_text) + status_text
```

- [ ] **Step 3: Pass custom config path to CoordinatorServer**

In `my_a2a/src/openharness_a2a/coordinator/server.py`, modify `__init__` to accept a `config_path` parameter and pass it to `AgentRegistry`:

```python
def __init__(
    self,
    host: str = "0.0.0.0",
    port: int = 8080,
    a2a_port: int = 8081,
    config_path: str | None = None,
) -> None:
    # ...
    self._agent_registry = AgentRegistry(
        static_config_path=Path(config_path) if config_path else None
    )
```

Update `create_server()` to forward `config_path`:

```python
def create_server(host: str, port: int, a2a_port: int, config_path: str | None = None) -> CoordinatorServer:
    return CoordinatorServer(host=host, port=port, a2a_port=a2a_port, config_path=config_path)
```

- [ ] **Step 4: Commit**

```bash
cd my_a2a && git add config/agents.yaml src/openharness_a2a/coordinator/router.py src/openharness_a2a/coordinator/server.py
git commit -m "feat(coordinator): add robot agents config, enhance RouterAgent, accept custom config path"
```

---

### Task 8: Modify GUI for Coordinator HTTP integration

**Files:**
- Modify: `CoMuRoS/CoMuRoS/chatty/chatty/chat_gui.py`
- Modify: `CoMuRoS/CoMuRoS/chatty/launch/chat_system.launch.py`

- [ ] **Step 1: Add `httpx` dependency to chatty setup.py**

Edit `CoMuRoS/chatty/setup.py`, add `'httpx'` to `install_requires` list:

```python
    install_requires=['setuptools', 'rclpy', 'httpx'],
```

- [ ] **Step 2: Add Coordinator HTTP client to ChatGUI**

The actual `chat_gui.py` has `send_message()` at line 348 and `append_text()` at line 184, with `Thread` already imported (line 16). Edit `chat_gui.py` to add httpx-based Coordinator communication.

Add the import at the top (near other imports, line 6-8 area):

```python
import httpx
```

In `ChatGUI.__init__()`, add after the existing `self.declare_parameter("config_file", ...)` block (around line 31-36):

```python
        # Coordinator URL for A2A mode
        self.declare_parameter("coordinator_url", "http://localhost:8080")
        self.coordinator_url = self.get_parameter("coordinator_url").get_parameter_value().string_value
        self._http_client = httpx.Client(timeout=60.0)
        self.get_logger().info(f"[ChatGUI] Coordinator URL: {self.coordinator_url}")
```

Modify `send_message()` (line 348) to also send to Coordinator:

```python
    def send_message(self, event=None):
        """Send user input via ROS /chat/input AND to A2A Coordinator."""
        user_input = self.entry.get().strip()
        if user_input:
            # 1. ROS publish (existing behavior, unchanged)
            out_msg = String()
            out_msg.data = f"human|{user_input}"
            self.publisher.publish(out_msg)
            self.get_logger().info(f"[ChatGUI] Sent -> {user_input}")

            # 2. Send to A2A Coordinator in background thread
            t = Thread(target=self._send_to_coordinator, args=(user_input,), daemon=True)
            t.start()

        self.entry.delete(0, 'end')

    def _send_to_coordinator(self, user_text: str):
        """Send user input to Coordinator and poll for result, then display."""
        import time
        try:
            response = self._http_client.post(
                f"{self.coordinator_url}/tasks",
                json={"prompt": user_text},
            )
            if response.status_code != 200:
                return
            task_id = response.json().get("task_id")
            # Poll for result (max 60s)
            for _ in range(60):
                time.sleep(1.0)
                poll = self._http_client.get(f"{self.coordinator_url}/tasks/{task_id}")
                if poll.status_code == 200:
                    data = poll.json()
                    status = data.get("status", "")
                    if status in ("completed", "failed"):
                        result_text = str(data.get("result", ""))
                        display_line = f"Coordinator|{result_text}"
                        self.window.after(0, lambda: self.on_output_direct(display_line))
                        return
            # Timeout
            self.window.after(0, lambda: self.on_output_direct("Coordinator|[Timeout] Task took too long"))
        except Exception as e:
            self.get_logger().error(f"Coordinator error: {e}")
            self.window.after(0, lambda: self.on_output_direct(f"Coordinator|[Error] {e}"))

    def on_output_direct(self, line: str):
        """Display a message directly in chat, bypassing ROS subscription.
        Strips timestamp prefix like '[2026-05-28 12:00:00]' if present,
        preserving tags like '[Error]' in the message body."""
        import re
        # Strip leading timestamp like "[2026-05-28 12:00:00] " only
        line = re.sub(r'^\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\]\s*', '', line)
        self.append_text(line)
```

- [ ] **Step 3: Add coordinator_url launch parameter**

Edit `chat_system.launch.py`. Add a `DeclareLaunchArgument` for `coordinator_url` near the existing launch arguments (`model`, `config_file`, etc.), and add a `parameters` list to the `chat_interface_node` Node definition.

Find the `chat_interface_node` Node (approximately line 63-70):

```python
    chat_interface_node = Node(
        package='chatty',
        executable='chat_gui',
        name='human_gui',
        output='screen',
    )
```

Change it to:

```python
    chat_interface_node = Node(
        package='chatty',
        executable='chat_gui',
        name='human_gui',
        output='screen',
        parameters=[{
            'coordinator_url': LaunchConfiguration('coordinator_url'),
            'config_file': LaunchConfiguration('config_file'),
        }],
    )
```

Near the other `DeclareLaunchArgument` calls (lines ~38-42), add:

```python
    DeclareLaunchArgument(
        'coordinator_url',
        default_value='http://localhost:8080',
        description='A2A Coordinator URL',
    ),
```

- [ ] **Step 4: Commit**

```bash
cd CoMuRoS && git add CoMuRoS/chatty/chatty/chat_gui.py CoMuRoS/chatty/setup.py CoMuRoS/chatty/launch/chat_system.launch.py
git commit -m "feat(gui): add httpx dep and Coordinator HTTP integration with polling"
```

---

### Task 9: Disable task_manager and update launch files

**Files:**
- Modify: `CoMuRoS/CoMuRoS/chatty/chatty/task_manager.py`
- Modify: `CoMuRoS/CoMuRoS/chatty/launch/chat_system.launch.py`

- [ ] **Step 1: Guard task_manager main() with environment variable**

In `task_manager.py`'s `main()`, add an early-exit so it can be disabled without deleting the file:

```python
def main():
    import os
    if os.environ.get("USE_A2A_COORDINATOR", "").lower() in ("1", "true", "yes"):
        print("[TaskManager] A2A Coordinator mode active — TaskManager disabled.")
        return
    # ... existing rclpy.init() and rest of main() unchanged ...
```

- [ ] **Step 2: Set env var in launch file for task_manager node**

In `chat_system.launch.py`, find the task_manager Node definition and add:

```python
Node(
    package='chatty',
    executable='task_manager',
    name='task_manager',
    env_vars={'USE_A2A_COORDINATOR': '1'},
    # ... other params unchanged ...
),
```

- [ ] **Step 3: Commit**

```bash
cd CoMuRoS && git add CoMuRoS/chatty/chatty/task_manager.py CoMuRoS/chatty/launch/chat_system.launch.py
git commit -m "feat(chatty): add A2A Coordinator mode toggle for task_manager"
```

---

### Task 10: Integration tests and launch verification

- [ ] **Step 1: Write integration test in my_a2a (with path setup)**

Create `my_a2a/tests/integration/conftest.py`:

```python
"""Path setup for cross-repo test imports from CoMuRoS."""
import os
import sys

_COMUROS_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "CoMuRoS", "CoMuRoS")
)
if _COMUROS_PATH not in sys.path:
    sys.path.insert(0, _COMUROS_PATH)

_MY_A2A_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "src")
)
if _MY_A2A_PATH not in sys.path:
    sys.path.insert(0, _MY_A2A_PATH)

_MINI_AGENT_PATH = os.path.join(_MY_A2A_PATH, "Mini-Agent")
if _MINI_AGENT_PATH not in sys.path:
    sys.path.insert(0, _MINI_AGENT_PATH)
```

Create `my_a2a/tests/integration/test_robot_worker.py`:

```python
"""Test: Robot Worker Tool execution."""

import pytest
from unittest.mock import MagicMock


@pytest.mark.asyncio
async def test_cleaning_bot_tool_execution():
    """Verify cleaning bot Tool is callable and delegates to node."""
    from cleaning_bot.tools import CleanTool
    mock_node = MagicMock()
    tool = CleanTool(node=mock_node)
    result = await tool.execute()
    assert result.success
    mock_node.clean.assert_called_once()


@pytest.mark.asyncio
async def test_delivery_bot_tool_execution():
    """Verify delivery bot DeliverFood Tool with parameters."""
    from delivery_bot.tools import DeliverFoodTool
    mock_node = MagicMock()
    tool = DeliverFoodTool(node=mock_node)
    result = await tool.execute(stall_number=1, table_number=3)
    assert result.success
    mock_node.deliver_food.assert_called_once_with(1, 3)


@pytest.mark.asyncio
async def test_drone_hover_tool_execution():
    """Verify drone Hover Tool with parameters."""
    from drone.tools import HoverTool
    mock_node = MagicMock()
    tool = HoverTool(node=mock_node)
    result = await tool.execute(x=1.0, y=2.0, z=3.0, yaw_deg=0.0)
    assert result.success
    mock_node.hover.assert_called_once_with(x=1.0, y=2.0, z=3.0, yaw_deg=0.0)


@pytest.mark.asyncio
async def test_robot_arm_pick_tool_execution():
    """Verify robot arm PickObject Tool."""
    from robot_llm.tools import PickObjectTool
    mock_node = MagicMock()
    mock_node.pick_object.return_value = True
    tool = PickObjectTool(node=mock_node)
    result = await tool.execute(object_name="green object")
    assert result.success
    mock_node.pick_object.assert_called_once_with("green object")


@pytest.mark.asyncio
async def test_tool_returns_failure_on_exception():
    """Verify Tool returns failure result when node raises."""
    from cleaning_bot.tools import CleanTool
    mock_node = MagicMock()
    mock_node.clean.side_effect = RuntimeError("Motor failure")
    tool = CleanTool(node=mock_node)
    result = await tool.execute()
    assert not result.success
    assert "Motor failure" in result.error
```

- [ ] **Step 2: Run tests**

```bash
cd my_a2a && python -m pytest tests/integration/test_robot_worker.py -v
```

Expected: 5 tests pass.

- [ ] **Step 3: Verify Coordinator starts and loads config**

```bash
cd my_a2a && timeout 5 python -m openharness_a2a.coordinator.cli 2>&1 | head -10 || true
```

Expected: No import errors; Coordinator starts.

- [ ] **Step 4: Commit**

```bash
cd my_a2a && git add tests/integration/conftest.py tests/integration/test_robot_worker.py
git commit -m "test: add robot Worker Tool integration tests with cross-repo path setup"
```

---

## Implementation Summary

| Phase | Task | Files |
|-------|------|-------|
| Prep | 1. Enhance MiniAgentAdapter | `mini_agent_adapter.py`, `a2a_server.py` (my_a2a) |
| Prep | 2. A2AWorkerNode base (with WS) | `a2a_worker_node.py` (CoMuRoS/robot_llm) |
| Workers | 3. cleaning_bot | `tools.py`, `cleaning_bot_llm.py` (rewrite) |
| Workers | 4. delivery_bot | `tools.py`, `delivery_bot_llm.py` (rewrite) |
| Workers | 5. drone | `tools.py`, `drone_llm.py` (rewrite) |
| Workers | 6. robot_arm | `tools.py`, `robot_llm.py` (rewrite) |
| Coordinator | 7. Config + RouterAgent | `agents.yaml`, `router.py`, `server.py` |
| GUI | 8. HTTP integration | `chat_gui.py`, `chat_system.launch.py` |
| Cleanup | 9. Disable task_manager | `task_manager.py`, launch files |
| Verify | 10. Integration tests | `test_robot_worker.py` |
