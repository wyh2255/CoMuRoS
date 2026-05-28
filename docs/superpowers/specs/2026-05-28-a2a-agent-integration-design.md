# CoMuRoS × my_a2a Agent 集成设计

将 CoMuRoS 内部的 LLM 实现替换为 A2A Agent 版本，task_manager 替换为 A2A Coordinator，robot LLM 节点替换为 A2A Worker（Mini-Agent + Tool Calling）。

## 1. 总体架构

```
┌─────────────────────────────────────────────────────────┐
│  my_a2a (独立进程)                                      │
│                                                         │
│  Coordinator (FastAPI :8080)                            │
│  ├─ RouterAgent (LLM 任务分解)                         │
│  ├─ AgentRegistry (Worker 能力注册表)                   │
│  └─ TaskQueue (任务生命周期管理)                         │
│       │                                                 │
│       │ A2A JSON-RPC (tasks/send)                       │
│       ▼                                                 │
└───────┬─────────────────┬─────────────────┬─────────────┘
        │                 │                 │
        │  Worker ── WS ── Coordinator (注册 + 心跳)      │
        │                 │                 │
        ▼                 ▼                 ▼
┌───────────────┐ ┌───────────────┐ ┌───────────────┐
│ Worker (ROS)  │ │ Worker (ROS)  │ │ Worker (ROS)  │
│ cleaning_bot  │ │ delivery_bot  │ │     drone     │ ...
│               │ │               │ │               │
│ Mini-Agent    │ │ Mini-Agent    │ │ Mini-Agent    │
│ + Tools       │ │ + Tools       │ │ + Tools       │
│     │         │ │     │         │ │     │         │
│     ▼         │ │     ▼         │ │     ▼         │
│ ROS services  │ │ ROS services  │ │ ROS services  │
└───────────────┘ └───────────────┘ └───────────────┘
        │                 │                 │
        └─────────────────┼─────────────────┘
                          │ ROS topics (status)
                          ▼
                    ┌──────────┐
                    │   GUI    │
                    │(customtk)│
                    └──────────┘
```

**集成方式**：方案 2 - Coordinator 独立进程，Worker 嵌入 ROS 节点。

**数据流**：用户输入 → GUI → HTTP POST /tasks → Coordinator (RouterAgent LLM 分解) → A2A tasks/send → Worker Mini-Agent (Tool Calling) → ROS service → Gazebo。Worker 启动时通过 WebSocket (`ws://coordinator:8080/ws/worker/{id}`) 向 Coordinator 注册，Coordinator 从 Worker 拉取 AgentCard 获取能力信息。状态反馈通过 ROS topic → GUI。

## 2. Worker 注册与发现流程

每个 Worker 启动时执行以下注册流程（同 `my_a2a/src/openharness_a2a/worker/coordinator_client.py` 的模式）：

1. Worker 启动 A2A HTTP Server（如 :8090）
2. Worker 通过 WebSocket 连接 `ws://coordinator:8080/ws/worker/{robot_name}`
3. Worker 发送 `WS_REGISTER` 消息，携带 `a2a_endpoint`
4. Coordinator 收到注册后，从 `{a2a_endpoint}/.well-known/agent-card.json` 拉取 AgentCard
5. Coordinator 解析 AgentCard 中的 `skills`，提取 `capabilities`、`backend`、`model`
6. Worker 每 30 秒发送 `WS_HEARTBEAT` 保持在线状态

AgentCard 的 `skills` 字段编码了机器人能力：
```json
{
  "skills": [
    {"id": "cleaning", "name": "cleaning", "description": "Capability: cleaning", "tags": ["cleaning"]},
    {"id": "backend", "name": "Backend", "description": "Execution backend: mini_agent", "tags": ["metadata", "backend", "mini_agent"]},
    {"id": "model", "name": "Model", "description": "LLM model: deepseek-v4-flash", "tags": ["metadata", "model", "deepseek-v4-flash"]}
  ]
}
```

## 3. 被替换的代码

| 文件 | 去掉 | 原因 |
|------|------|------|
| `task_manager.py` (整个文件) | ~1745 行 | Coordinator + RouterAgent 替代 |
| `cleaning_bot_llm.py` | `generate_action_prompt()`, `execute_python_code()`, `execute_task()`, `TestOption`/`option_list` | Mini-Agent Tool Calling 替代 |
| `drone_llm.py` | 同上 | 同上 |
| `delivery_bot_llm.py` | 同上 | 同上 |
| `robot_llm.py` | 同上 | 同上 |
| 所有 robot_llm | `call_openai()` / `call_ollama()` / `call_gemini()` / `call_xai()` | LLM 调用由 Mini-Agent 统一管理 |

**保留的代码**：`goto_service()`、`clean()`、`deliver_food()`、`hover()`、`pick_object()` 等机器人核心动作，改为注册为 Mini-Agent Tool。状态管理（`update_robot_state()`、`robot_task_completed()` 等）、ROS 订阅/发布逻辑全部保留。

## 4. Worker 节点设计

每个 Worker 是 ROS 2 Node + A2A HTTP Server + Mini-Agent + WebSocket 客户端 的融合进程：

```
Worker（单进程）
├─ 主线程：rclpy.spin()           ← ROS 订阅/发布
├─ 后台线程 1：uvicorn            ← A2A HTTP Server（如 :8090）
├─ 后台线程 2：asyncio            ← WebSocket 注册 + 心跳
├─ Mini-Agent 实例                ← LLM + Tool Calling
│   └─ Tools（注册为 Mini-Agent function）
│       ├─ cleaning_bot: clean()
│       ├─ delivery_bot: deliver_food(stall, table), clear_table(table, food)
│       ├─ drone: hover(x, y, z, yaw), describe_scene(prompt)
│       └─ robot_arm: pick_object(name)
└─ 保留：任务状态发布、机器人状态发布、聊天历史记录
```

每个 Tool 直接调用现有 ROS 服务逻辑，不再通过 exec() 执行生成的代码。Tool 函数负责状态更新和返回值。

**System prompt 传递路径**：子类覆写 `_get_system_prompt()` → `start_a2a_server()` 传递给 `create_worker_a2a_server()` → `create_worker_a2a_server()` 传递给 `MiniAgentAdapter(system_prompt=...)` → `MiniAgentAdapter._build_agent()` 注入 Agent。

## 5. Coordinator 设计

Coordinator 以 `openharness-a2a` 独立进程运行。Worker 通过 WebSocket 注册（见 §2），Coordinator 从 AgentCard 自动发现能力。同时也支持 `agents.yaml` 静态配置作为备用。

**RouterAgent 功能需求**：
- 支持 sequential / parallel / single 执行顺序（对应 Plan / Independent Tasks / Single）
- 机器人状态感知（通过 Worker 心跳追踪在线状态）
- 事件驱动的 re-planning（接收事件消息后重新 route）
- STOP / resume 通过 A2A cancel() + 重新下发任务实现
- System prompt 包含餐厅环境信息（stalls, tables, sink, food items）

## 6. GUI 改动

**chat_gui.py 改动**：
- 在现有 `send_message()` 方法中添加 HTTP POST 到 Coordinator `/tasks`
- 轮询 `/tasks/{task_id}` 获取结果（轮询间隔 1s，最多 60s）
- Coordinator 返回的结果通过现有 `append_text()` 渲染为 "Coordinator" 消息
- `coordinator_url` 作为 ROS 参数，默认 `http://localhost:8080`
- 保留：ROS 订阅 `/chat/output`、`/chat/task_status`、customtkinter 界面全部不变

**chat_manager.py**：不改动。

**启动步骤变化**：
```bash
# Terminal 1: 仿真 + robots（去掉 task_manager）
ros2 launch multi_robot multi_robot.launch.py
ros2 launch robot_llm robot_llms.launch.py
ros2 launch robot_llm robot_services.launch.py

# Terminal 2: GUI
ros2 launch chatty chat_system.launch.py

# Terminal 3: Coordinator
cd my_a2a && openharness-a2a
```

## 7. 错误处理

- **Worker 离线**：Coordinator 收到 WS 心跳超时，标记 Worker offline，RouterAgent 重新规划用替代 Worker
- **Tool 执行失败**：Tool 返回 `ToolResult(success=False, error=...)` → Mini-Agent 感知 → step_callback 推状态 → Coordinator 决定是否 re-plan
- **Coordinator 崩溃**：Worker WebSocket 断开后自动重连（最多 3 次），GUI 显示轮询超时提示
- **LLM 调用失败**：Mini-Agent 内部重试机制（已有），失败后 Worker A2A 任务标记为 failed
- **sys.path 降级**：Worker 通过 `MY_A2A_PATH` 和 `MINI_AGENT_PATH` 环境变量定位依赖，未设置则 fallback 到相对路径

## 8. 测试策略

- **单元测试**：Tool 函数独立测试（Mock ROS node，验证 Tool.execute() 调用对应方法）
- **集成测试**：A2A task 发送 → Worker 接收 → Tool 执行（Mock ROS services）
- **端到端测试**：GUI → Coordinator → Worker → ROS service（全链路，需要 Gazebo 或 mock）
- **回归测试**：现有 ROS launch 流程是否正常工作，保留的核心功能无退化

## 9. 实施顺序

1. **Phase 1**：MiniAgentAdapter 增强 + A2AWorkerNode 基类（含 WS 注册）
2. **Phase 2**：Worker 改造一个 robot（如 cleaning_bot），验证 Tool Calling + WS 注册模式
3. **Phase 3**：迁移其余 3 个 robot Worker
4. **Phase 4**：Coordinator 配置 + RouterAgent 增强
5. **Phase 5**：GUI HTTP 集成 + 禁用 task_manager
