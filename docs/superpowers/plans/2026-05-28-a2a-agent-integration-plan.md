# A2A Agent Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace all hard-coded LLM API calls in CoMuRoS with A2A Agent-based architecture: Coordinator replaces task_manager.py, each robot node becomes an A2A Worker with Mini-Agent + Tool Calling.

**Architecture:** Coordinator (independent process, FastAPI :8080) performs LLM-driven task decomposition via RouterAgent. Four robot Workers (ROS nodes with embedded A2A HTTP server + Mini-Agent) execute tasks via Tool Calling → ROS services. GUI sends user input to Coordinator via HTTP POST.

**Tech Stack:** ROS 2 (Python), FastAPI + uvicorn, A2A SDK (>1.0.0), Mini-Agent (MiniMax), DeepSeek API

---

### Task 1: Enhance MiniAgentAdapter to accept custom tools

**Files:**
- Modify: `my_a2a/src/openharness_a2a/worker/mini_agent_adapter.py`
- Modify: `my_a2a/src/openharness_a2a/worker/a2a_server.py`

- [ ] **Step 1: Add `extra_tools` parameter to MiniAgentAdapter**

Edit `my_a2a/src/openharness_a2a/worker/mini_agent_adapter.py`, add `extra_tools` parameter to `__init__` and wire it into `_build_agent()`:

```python
# In __init__, add parameter after system_prompt:
def __init__(
    self,
    model: str = "MiniMax-M2.7",
    system_prompt: str = "",
    max_steps: int = 50,
    workspace_dir: str = "./workspace",
    extra_tools: list | None = None,  # NEW
):
    self._model = model
    self._system_prompt = system_prompt or ""
    self._max_steps = max_steps
    self._workspace_dir = Path(workspace_dir)
    self._agent: Agent | None = None
    self._extra_tools = extra_tools or []  # NEW
```

```python
# In _build_agent(), after building the tools list and before creating Agent, append extra tools:
            tools = [
                ReadTool(),
                WriteTool(),
                BashTool(workspace_dir=config.agent.workspace_dir),
            ]
            # NEW: append custom tools
            if self._extra_tools:
                tools.extend(self._extra_tools)
```

Do the same in the `if config is None:` fallback path:

```python
            tools = [ReadTool(), WriteTool()]
            # NEW: append custom tools
            if self._extra_tools:
                tools.extend(self._extra_tools)
```

- [ ] **Step 2: Accept extra_tools in create_worker_a2a_server**

Edit `my_a2a/src/openharness_a2a/worker/a2a_server.py`, add `extra_tools` parameter:

```python
def create_worker_a2a_server(
    worker_id: str,
    host: str = "0.0.0.0",
    port: int = 8090,
    capabilities: list[str] | None = None,
    backend: str = "openharness",
    model: str = "claude-opus-4-5",
    extra_tools: list | None = None,  # NEW
) -> uvicorn.Server:
```

And pass it through when creating the MiniAgentAdapter:

```python
    if backend == "mini_agent":
        executor = MiniAgentAdapter(extra_tools=extra_tools)
```

- [ ] **Step 3: Run existing tests to verify no regression**

```bash
cd my_a2a && python -m pytest tests/ -v --timeout=30 2>&1 | tail -30
```

Expected: existing tests still pass.

- [ ] **Step 4: Commit**

```bash
cd my_a2a && git add src/openharness_a2a/worker/mini_agent_adapter.py src/openharness_a2a/worker/a2a_server.py
git commit -m "feat(worker): add extra_tools parameter to MiniAgentAdapter"
```

---

### Task 2: Create base A2AWorkerNode class in CoMuRoS

**Files:**
- Create: `CoMuRoS/CoMuRoS/robot_llm/robot_llm/a2a_worker_node.py`
- Modify: `CoMuRoS/CoMuRoS/robot_llm/setup.py`

- [ ] **Step 1: Create the base A2AWorkerNode**

Create `CoMuRoS/CoMuRoS/robot_llm/robot_llm/a2a_worker_node.py`:

```python
#!/usr/bin/env python3
"""Base ROS 2 Node with embedded A2A Worker + Mini-Agent."""

import json
import os
import sys
import threading
import time

import uvicorn
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from ament_index_python.packages import get_package_share_directory


class TaskCancelledException(Exception):
    pass


class A2AWorkerNode(Node):
    """Base ROS Node that runs an embedded A2A HTTP server with Mini-Agent.

    Subclasses must define:
      - ROBOT_NAME, ROBOT_TYPE, PACKAGE_NAME (class constants or instance attrs)
      - _get_tools() -> list  (return list of Mini-Agent Tool instances)
      - _get_a2a_port() -> int
    """

    def __init__(self, node_name: str, robot_name: str, package_name: str, port: int):
        super().__init__(node_name)

        self.robot_name = robot_name
        self._package_name = package_name
        self._a2a_port = port
        self.current_time = "Hours: 00, Minutes: 10, Seconds: 00"
        self.robot_task = ""
        self.robot_states = {}

        self._task_cancelled = False

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
            String, "/robot_states", self.on_robot_states, 10,
            callback_group=self.seq_group,
        )
        self.current_time_sub = self.create_subscription(
            String, "/current_time", self.on_current_time, 10,
            callback_group=self.seq_group,
        )
        self.sub_tasks_json = self.create_subscription(
            String, "/task_manager/tasks_json", self.on_tasks_json, 10,
            callback_group=self.multi_group,
        )
        self.sub_chat_output = self.create_subscription(
            String, "/chat/output", self.on_chat_output, 10,
            callback_group=self.single_group,
        )
        self.sub_chat_task_status = self.create_subscription(
            String, "/chat/task_status", self.on_chat_task_status, 10,
        )

        # File paths
        directry = "data"
        package_path = get_package_share_directory(package_name)
        self.history_file = os.path.join(package_path, directry, f"{robot_name}_chat_history.txt")
        self.robot_task_history = os.path.join(package_path, directry, f"{robot_name}_task_history.txt")

        self._a2a_server = None
        self._a2a_thread = None

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

    # --- A2A Server ---
    def start_a2a_server(self):
        """Start A2A HTTP server in a background thread."""
        from openharness_a2a.worker.a2a_server import create_worker_a2a_server

        extra_tools = self._get_tools()
        server = create_worker_a2a_server(
            worker_id=self.robot_name,
            host="0.0.0.0",
            port=self._a2a_port,
            capabilities=self._get_capabilities(),
            backend="mini_agent",
            model=self._get_model(),
            extra_tools=extra_tools,
        )
        self._a2a_server = server

        async def _serve():
            await server.serve()

        def _run():
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(_serve())

        self._a2a_thread = threading.Thread(target=_run, daemon=True)
        self._a2a_thread.start()
        self.get_logger().info(f"A2A server started on port {self._a2a_port}")

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
        """Handle incoming task JSON from legacy task_manager (retained for backward compat).
        In A2A mode, tasks arrive via the A2A MiniAgentAdapter instead."""
        pass  # Override in subclass if dual-mode needed

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

    # --- Lifecycle ---
    def destroy_node(self):
        self.get_logger().info("Shutting down A2A worker...")
        super().destroy_node()
```

- [ ] **Step 2: Update robot_llm setup.py to export new module**

Edit `CoMuRoS/CoMuRoS/robot_llm/setup.py`, ensure `a2a_worker_node` is included in `py_modules` or `packages`:

```python
# In setup(), verify the packages list includes robot_llm
# (if using find_packages, it's already covered)
```

- [ ] **Step 3: Commit**

```bash
cd CoMuRoS && git add CoMuRoS/robot_llm/robot_llm/a2a_worker_node.py && git commit -m "feat(robot_llm): add base A2AWorkerNode class"
```

---

### Task 3: Create cleaning_bot tools and A2A Worker

**Files:**
- Create: `CoMuRoS/CoMuRoS/cleaning_bot/cleaning_bot/tools.py`
- Modify: `CoMuRoS/CoMuRoS/cleaning_bot/cleaning_bot/cleaning_bot_llm.py`
- Modify: `CoMuRoS/CoMuRoS/cleaning_bot/setup.py`

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

- [ ] **Step 2: Refactor cleaning_bot_llm.py to extend A2AWorkerNode**

Modify `CoMuRoS/CoMuRoS/cleaning_bot/cleaning_bot/cleaning_bot_llm.py`:

```python
#!/usr/bin/env python3
"""Cleaning bot A2A Worker node."""

import sys
import os

# Add my_a2a to sys.path (adjust path as needed)
_A2A_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "my_a2a", "src")
if _A2A_PATH not in sys.path:
    sys.path.insert(0, _A2A_PATH)

# Add Mini-Agent to sys.path
_MINI_AGENT_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "my_a2a", "src", "Mini-Agent")
if _MINI_AGENT_PATH not in sys.path:
    sys.path.insert(0, _MINI_AGENT_PATH)

import time
import subprocess

import rclpy
from rclpy.executors import MultiThreadedExecutor
from robot_interface.srv import GotoPoseHolonomic
from std_msgs.msg import Bool  # noqa: F811

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
        self.robot_type = ROBOT_TYPE

        # GoTo service client
        self._goto_client = self.create_client(
            GotoPoseHolonomic, "/r1/goto_pose", callback_group=self.multi_group
        )
        self._cancel_goto_pub = self.create_publisher(Bool, "/r1/cancel_goto_pose_goal", 10)

        # Start A2A server
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

    # --- ROS service wrappers (retained from original) ---
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

- [ ] **Step 3: Update cleaning_bot setup.py**

Edit `CoMuRoS/CoMuRoS/cleaning_bot/setup.py`, add `tools` module to package list and add `robot_llm` dependency:

```python
# In packages, ensure cleaning_bot is included
# Add to install_requires if needed: 'robot_llm'
```

- [ ] **Step 4: Commit**

```bash
cd CoMuRoS && git add CoMuRoS/cleaning_bot/cleaning_bot/tools.py CoMuRoS/cleaning_bot/cleaning_bot/cleaning_bot_llm.py
git commit -m "feat(cleaning_bot): refactor to A2A Worker with CleanTool"
```

---

### Task 4: Create delivery_bot tools and A2A Worker

**Files:**
- Create: `CoMuRoS/CoMuRoS/delivery_bot/delivery_bot/tools.py`
- Modify: `CoMuRoS/CoMuRoS/delivery_bot/delivery_bot/delivery_bot_llm.py`

- [ ] **Step 1: Create delivery_bot tools**

Create `CoMuRoS/CoMuRoS/delivery_bot/delivery_bot/tools.py`:

```python
"""Mini-Agent Tool definitions for delivery_bot."""

import time
import subprocess

from mini_agent.tools.base import Tool, ToolResult

# Location constants
TableLocation = {
    1: [0.0, -1.3, 0.0], 2: [3.0, -1.3, 0.0],
    3: [6.0, -1.3, 0.0], 4: [9.0, -1.3, 0.0],
}
StallLocation = {
    1: [0.0, 0.0, 0.0], 2: [4.0, 0.0, 0.0], 3: [8.0, 0.0, 0.0],
}
food = ["food1", "food2", "food3"]


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
            return ToolResult(
                success=True,
                content=f"Delivered {food[stall_number-1]} from stall {stall_number} to table {table_number}.",
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
            "Check chat history to determine which food was delivered to the table. "
            "Navigates: home -> table -> sink -> home."
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

- [ ] **Step 2: Refactor delivery_bot_llm.py**

Modify `CoMuRoS/CoMuRoS/delivery_bot/delivery_bot/delivery_bot_llm.py` similarly to cleaning_bot:
- Extend `A2AWorkerNode` instead of `Node`
- Robot name `delivery_bot`, port `8091`
- `_get_tools()` returns `[DeliverFoodTool(node=self), ClearTableTool(node=self)]`
- `_get_capabilities()` returns `["food_delivery", "table_clearing"]`
- Keep `goto_service()`, `teleport()`, `deliver_food()`, `clear_table()` methods
- Remove: `generate_action_prompt()`, `execute_task()`, `execute_python_code()`, `TestOption`/`option_list`, all `call_openai()` etc.

- [ ] **Step 3: Commit**

```bash
cd CoMuRoS && git add CoMuRoS/delivery_bot/delivery_bot/tools.py CoMuRoS/delivery_bot/delivery_bot/delivery_bot_llm.py
git commit -m "feat(delivery_bot): refactor to A2A Worker with DeliverFood and ClearTable Tools"
```

---

### Task 5: Create drone tools and A2A Worker

**Files:**
- Create: `CoMuRoS/CoMuRoS/drone/drone/tools.py`
- Modify: `CoMuRoS/CoMuRoS/drone/drone/drone_llm.py`

- [ ] **Step 1: Create drone tools**

Write `CoMuRoS/CoMuRoS/drone/drone/tools.py` with two Tools:

```python
"""Mini-Agent Tool definitions for drone."""

from mini_agent.tools.base import Tool, ToolResult


class HoverTool(Tool):
    """Move drone to a position."""

    def __init__(self, node):
        self._node = node

    @property
    def name(self) -> str:
        return "hover"

    @property
    def description(self) -> str:
        return "Move drone to specified 3D position. Coordinates in meters, yaw in degrees."

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "x": {"type": "number", "description": "Target X coordinate"},
                "y": {"type": "number", "description": "Target Y coordinate"},
                "z": {"type": "number", "description": "Target Z (altitude in meters)", "default": 2.0},
                "yaw_deg": {"type": "number", "description": "Target yaw angle in degrees", "default": 0.0},
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
    """Describe what the drone sees."""

    def __init__(self, node):
        self._node = node

    @property
    def name(self) -> str:
        return "describe_scene"

    @property
    def description(self) -> str:
        return "Describe the scene from the drone's bottom camera. Ask a question about what the drone sees."

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
            self._node.describe_screen(prompt)
            return ToolResult(success=True, content=f"Scene query completed: {prompt}")
        except Exception as e:
            return ToolResult(success=False, content="", error=str(e))
```

- [ ] **Step 2: Refactor drone_llm.py**

Modify to extend `A2AWorkerNode`. Robot name `drone`, port `8092`. Keep `hover()`, `goto_service()`, `describe_screen()`, `query_callback()`, `image_callback()`. Remove LLM code generation functions.

- [ ] **Step 3: Commit**

```bash
cd CoMuRoS && git add CoMuRoS/drone/drone/tools.py CoMuRoS/drone/drone/drone_llm.py
git commit -m "feat(drone): refactor to A2A Worker with Hover and DescribeScene Tools"
```

---

### Task 6: Create robot_arm tools and A2A Worker

**Files:**
- Create: `CoMuRoS/CoMuRoS/robot_llm/robot_llm/tools.py`
- Modify: `CoMuRoS/CoMuRoS/robot_llm/robot_llm/robot_llm.py`

- [ ] **Step 1: Create robot arm Tool**

Create `CoMuRoS/CoMuRoS/robot_llm/robot_llm/tools.py` with a `PickObjectTool`:

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
        return "Pick an object by color. Valid object names: 'green object', 'brown object', 'grey object'."

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

- [ ] **Step 2: Refactor robot_llm.py**

Modify to extend `A2AWorkerNode`. Robot name `robot1`, port `8093`. `_get_tools()` returns `[PickObjectTool(node=self)]`. `_get_capabilities()` returns `["pick_and_place"]`. Keep `pick_object()`, pick_* wrappers. Remove LLM code generation functions.

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

Replace the content of `my_a2a/config/agents.yaml`:

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

- [ ] **Step 2: Enhance RouterAgent system prompt for robot task planning**

Edit `my_a2a/src/openharness_a2a/coordinator/router.py`, replace `ROUTER_SYSTEM_PROMPT`:

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

- [ ] **Step 3: Update RouterAgent constructor to accept config path**

In `router.py`, add a class method or update `__init__` to accept an explicit config path so we can point it to the custom `agents.yaml`:

```python
def __init__(
    self,
    registry: AgentRegistry,
    llm_model: str = "claude-opus-4-5",
    config_path: str | None = None,  # NEW
) -> None:
    self._registry = registry
    self._llm_model = llm_model
    self._config_path = config_path  # NEW
```

- [ ] **Step 4: Commit**

```bash
cd my_a2a && git add config/agents.yaml src/openharness_a2a/coordinator/router.py
git commit -m "feat(coordinator): add robot agents config and enhance RouterAgent for task planning"
```

---

### Task 8: Modify GUI for Coordinator HTTP integration

**Files:**
- Modify: `CoMuRoS/CoMuRoS/chatty/chatty/chat_gui.py`
- Modify: `CoMuRoS/CoMuRoS/chatty/launch/chat_system.launch.py`

- [ ] **Step 1: Add Coordinator HTTP client to chat_gui.py**

In `chat_gui.py`, find the `send_message` method (or equivalent where user input is published to `/chat/input`). Add an HTTP call to the Coordinator before the ROS publish:

```python
import httpx
import threading
import json

class ChatGUI:
    def __init__(self, ...):
        # ... existing init ...
        self.declare_parameter("coordinator_url", "http://localhost:8080")
        self.coordinator_url = self.get_parameter("coordinator_url").value
        self._http_client = httpx.Client(timeout=60.0)

    def send_to_coordinator(self, user_text: str):
        """Send user input to A2A Coordinator and return plan response."""
        try:
            response = self._http_client.post(
                f"{self.coordinator_url}/tasks",
                json={"prompt": user_text},
            )
            if response.status_code == 200:
                task_id = response.json().get("task_id")
                # Poll for result
                for _ in range(60):  # max 60s wait
                    time.sleep(1.0)
                    poll = self._http_client.get(f"{self.coordinator_url}/tasks/{task_id}")
                    if poll.status_code == 200:
                        data = poll.json()
                        if data.get("status") in ("completed", "failed"):
                            return data
                return {"status": "timeout", "result": None}
        except Exception as e:
            self.get_logger().error(f"Coordinator error: {e}")
            return {"status": "error", "result": str(e)}

    def on_send(self, user_text: str):
        """Called when user presses send."""
        # 1. Display user message immediately
        self.display_message("Human", user_text)

        # 2. Send to Coordinator in background thread
        def _coordinator_send():
            result = self.send_to_coordinator(user_text)
            # Display result in GUI
            self.after(0, lambda: self.display_coordinator_result(result))

        threading.Thread(target=_coordinator_send, daemon=True).start()

        # 3. Also publish to ROS /chat/input for history (backward compat)
        msg = String()
        msg.data = f"human|{user_text}"
        self.pub_chat_input.publish(msg)

    def display_coordinator_result(self, result: dict):
        """Display Coordinator response in the chat window."""
        status = result.get("status", "unknown")
        response = result.get("result", "")
        if status == "completed":
            self.display_message("Coordinator", str(response))
        elif status == "error":
            self.display_message("Coordinator", f"[Error] {response}")
        elif status == "timeout":
            self.display_message("Coordinator", "[Timeout] Task took too long")
```

- [ ] **Step 2: Add coordinator_url launch parameter**

Edit `chat_system.launch.py`, add `coordinator_url` parameter with default `http://localhost:8080`:

```python
# Add to launch arguments:
DeclareLaunchArgument('coordinator_url', default_value='http://localhost:8080')
```

- [ ] **Step 3: Commit**

```bash
cd CoMuRoS && git add CoMuRoS/chatty/chatty/chat_gui.py CoMuRoS/chatty/launch/chat_system.launch.py
git commit -m "feat(gui): add Coordinator HTTP integration"
```

---

### Task 9: Disable task_manager and update launch files

**Files:**
- Modify: `CoMuRoS/CoMuRoS/chatty/chatty/task_manager.py`
- Modify: `CoMuRoS/CoMuRoS/chatty/launch/chat_system.launch.py`

- [ ] **Step 1: Guard task_manager main() with environment variable**

Add an early-exit at the top of `task_manager.py`'s `main()` so it can be disabled without deleting:

```python
def main():
    import os
    if os.environ.get("USE_A2A_COORDINATOR", "").lower() in ("1", "true", "yes"):
        print("[TaskManager] A2A Coordinator mode active — TaskManager disabled.")
        return
    # ... existing main() code ...
```

- [ ] **Step 2: Set env var in launch file**

In `chat_system.launch.py`, add an env var to the task_manager node:

```python
# In the Node definition for task_manager:
Node(
    package='chatty',
    executable='task_manager',
    name='task_manager',
    # ... other params ...
    env_vars={'USE_A2A_COORDINATOR': '1'},  # NEW: disable when using A2A
),
```

- [ ] **Step 3: Update start script docs**

Update any README/CLAUDE.md references for the launch sequence:

```bash
# New terminal sequence:
# Terminal 1: Simulation + robots
ros2 launch multi_robot multi_robot.launch.py
ros2 launch robot_llm robot_llms.launch.py
ros2 launch robot_llm robot_services.launch.py

# Terminal 2: GUI (no task_manager)
ros2 launch chatty chat_system.launch.py

# Terminal 3: A2A Coordinator
cd my_a2a && openharness-a2a
```

- [ ] **Step 4: Commit**

```bash
cd CoMuRoS && git add CoMuRoS/chatty/chatty/task_manager.py CoMuRoS/chatty/launch/chat_system.launch.py && git commit -m "feat(chatty): add A2A Coordinator mode toggle for task_manager"
```

---

### Task 10: Clean up dead LLM code in robot nodes

**Files:**
- Modify: `CoMuRoS/CoMuRoS/cleaning_bot/cleaning_bot/cleaning_bot_llm.py`
- Modify: `CoMuRoS/CoMuRoS/delivery_bot/delivery_bot/delivery_bot_llm.py`
- Modify: `CoMuRoS/CoMuRoS/drone/drone/drone_llm.py`
- Modify: `CoMuRoS/CoMuRoS/robot_llm/robot_llm/robot_llm.py`

- [ ] **Step 1: Remove dead code from each robot_llm file**

For each of the 4 robot_llm files, remove:
- `import openai` and related imports (`from openai import OpenAI`)
- `openai.api_key = ...` assignments
- `generate_action_prompt()` method (LLM code generation)
- `execute_task()` method (manual prompt building)
- `execute_python_code()` standalone function (exec-based execution)
- `TestOption` dataclass and `option_list`
- `ROBOT_TYPE` constant (now in worker class or tools)

- [ ] **Step 2: Verify cleaned files still import correctly**

```bash
cd CoMuRoS && python -c "from cleaning_bot.cleaning_bot_llm import CleaningBotWorker; print('OK')"
```

- [ ] **Step 3: Commit**

```bash
cd CoMuRoS && git add CoMuRoS/cleaning_bot/ CoMuRoS/delivery_bot/ CoMuRoS/drone/ CoMuRoS/robot_llm/
git commit -m "refactor: remove dead LLM code from robot nodes (now handled by A2A/Mini-Agent)"
```

---

### Task 11: End-to-end integration test and launch verification

- [ ] **Step 1: Write integration test in my_a2a**

Create `my_a2a/tests/integration/test_robot_worker.py`:

```python
"""Test: Coordinator routes task to robot Worker, Worker executes via Tool Calling."""

import pytest
from unittest.mock import MagicMock, patch


@pytest.mark.asyncio
async def test_cleaning_bot_tool_execution():
    """Verify cleaning bot Tool is callable with correct parameters."""
    from cleaning_bot.tools import CleanTool
    mock_node = MagicMock()
    tool = CleanTool(node=mock_node)
    result = await tool.execute()
    assert result.success
    mock_node.clean.assert_called_once()


@pytest.mark.asyncio
async def test_delivery_bot_tool_execution():
    """Verify delivery bot DeliverFood Tool."""
    from delivery_bot.tools import DeliverFoodTool
    mock_node = MagicMock()
    tool = DeliverFoodTool(node=mock_node)
    result = await tool.execute(stall_number=1, table_number=3)
    assert result.success
    mock_node.deliver_food.assert_called_once_with(1, 3)
```

- [ ] **Step 2: Run tests**

```bash
cd my_a2a && python -m pytest tests/integration/test_robot_worker.py -v
```

Expected: Tests pass, verifying Tool interface works.

- [ ] **Step 3: Verify Coordinator starts and loads config**

```bash
cd my_a2a && openharness-a2a &
sleep 3
curl -s http://localhost:8080/health | python -m json.tool
kill %1
```

Expected: `{"status": "healthy"}`

- [ ] **Step 4: Commit**

```bash
cd my_a2a && git add tests/integration/test_robot_worker.py && git commit -m "test: add robot Worker Tool integration tests"
```

---

## Implementation Summary

| Phase | Task | Files |
|-------|------|-------|
| Prep | 1. Enhance MiniAgentAdapter | `mini_agent_adapter.py`, `a2a_server.py` (my_a2a) |
| Prep | 2. A2AWorkerNode base class | `a2a_worker_node.py` (CoMuRoS/robot_llm) |
| Workers | 3. cleaning_bot | `tools.py`, `cleaning_bot_llm.py` |
| Workers | 4. delivery_bot | `tools.py`, `delivery_bot_llm.py` |
| Workers | 5. drone | `tools.py`, `drone_llm.py` |
| Workers | 6. robot_arm | `tools.py`, `robot_llm.py` |
| Coordinator | 7. Config + RouterAgent | `agents.yaml`, `router.py` |
| GUI | 8. HTTP integration | `chat_gui.py`, `chat_system.launch.py` |
| Cleanup | 9. Disable task_manager | `task_manager.py`, launch files |
| Cleanup | 10. Remove dead LLM code | 4 robot_llm files |
| Verify | 11. Integration tests | `test_robot_worker.py` |
