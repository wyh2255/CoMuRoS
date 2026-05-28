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

**数据流**：用户输入 → GUI → HTTP POST /task → Coordinator (RouterAgent) → A2A tasks/send → Worker Mini-Agent (Tool Calling) → ROS service → Gazebo。状态反馈通过 ROS topic → GUI。

## 2. 被替换的代码

| 文件 | 去掉 | 原因 |
|------|------|------|
| `task_manager.py` (整个文件) | ~1745 行 | Coordinator + RouterAgent 替代 |
| `cleaning_bot_llm.py` | `generate_action_prompt()`, `execute_python_code()`, `execute_task()`, `TestOption`/`option_list` | Mini-Agent Tool Calling 替代 |
| `drone_llm.py` | 同上 | 同上 |
| `delivery_bot_llm.py` | 同上 | 同上 |
| `robot_llm.py` | 同上 | 同上 |
| 所有 robot_llm | `call_openai()` / `call_ollama()` / `call_gemini()` / `call_xai()` | LLM 调用由 Mini-Agent 统一管理 |

**保留的代码**：`goto_service()`、`clean()`、`deliver_food()`、`hover()`、`pick_object()` 等机器人核心动作，改为注册为 Mini-Agent Tool。状态管理（`update_robot_state()`、`robot_task_completed()` 等）、ROS 订阅/发布逻辑全部保留。

## 3. Worker 节点设计

每个 Worker 是 ROS 2 Node + A2A HTTP Server + Mini-Agent 的融合进程：

```
Worker（单进程）
├─ 主线程：rclpy.spin()           ← ROS 订阅/发布
├─ 后台线程：uvicorn              ← A2A HTTP Server（如 :8090）
├─ Mini-Agent 实例                ← LLM + Tool Calling
│   └─ Tools（注册为 Mini-Agent function）
│       ├─ cleaning_bot: clean()
│       ├─ delivery_bot: deliver_food(stall, table), clear_table(table, food)
│       ├─ drone: hover(x, y, z, yaw), describe_scene(prompt)
│       └─ robot_arm: pick_object(name)
└─ 保留：任务状态发布、机器人状态发布、聊天历史记录
```

每个 Tool 直接调用现有 ROS 服务逻辑，不再通过 exec() 执行生成的代码。Tool 函数负责状态更新和返回值。

## 4. Coordinator 设计

Coordinator 以 `openharness-a2a` 独立进程运行。RouterAgent 通过 agents.yaml 配置获取所有 Worker 能力信息。

**agents.yaml（新增 robot 条目）**：
```yaml
agents:
  - id: cleaning-bot
    description: "全向驱动机器人，沿预定路径清洁餐厅地面，能移除障碍物"
    endpoint: http://localhost:8090
    capabilities: [cleaning, obstacle_removal]
    backend: mini_agent
    model: deepseek-v4-flash

  - id: delivery-bot
    description: "差速驱动机器人，从摊位送餐到餐桌、清理餐桌餐具至水槽"
    endpoint: http://localhost:8091
    capabilities: [food_delivery, table_clearing]
    backend: mini_agent
    model: deepseek-v4-flash

  - id: drone
    description: "四旋翼无人机，悬停移动到指定位置、通过底部摄像头描述场景"
    endpoint: http://localhost:8092
    capabilities: [aerial_inspection, scene_description]
    backend: mini_agent
    model: deepseek-v4-flash

  - id: robot-arm
    description: "固定机械臂，拾取指定颜色的物体（绿/棕/灰）"
    endpoint: http://localhost:8093
    capabilities: [pick_and_place]
    backend: mini_agent
    model: deepseek-v4-flash
```

**RouterAgent 增强需求**：
- 支持 sequential / parallel / single 执行顺序（对应 Plan / Independent Tasks / Single）
- 机器人状态感知（通过 Worker 心跳 + WebSocket 推送）
- 事件驱动的 re-planning（接收事件消息后重新 route）
- STOP / resume 通过 A2A cancel() + 重新下发任务实现

## 5. GUI 改动

**chat_gui.py 改动**：
- 新增：HTTP client，用户输入后 POST 到 Coordinator `/task`
- 新增：显示 Coordinator 返回的 plan/response
- 新增：`model` 参数替换为 Coordinator 地址参数
- 保留：ROS 订阅 `/chat/output`、`/chat/task_status`
- 保留：customtkinter 界面组件全部不变

**chat_manager.py**：不改动。

**启动步骤变化**：
```bash
# Terminal 1: 仿真 + robots + GUI（去掉 task_manager）
ros2 launch multi_robot multi_robot.launch.py
ros2 launch robot_llm robot_llms.launch.py
ros2 launch chatty chat_system.launch.py  # 不再启动 task_manager

# Terminal 2: Coordinator（新增）
openharness-a2a
```

## 6. 错误处理

- **Worker 离线**：Coordinator 收到 WS 心跳超时，标记 Worker offline，RouterAgent 重新规划用替代 Worker
- **Tool 执行失败**：Tool 返回错误字符串 → Mini-Agent 感知 → step_callback 推状态 → Coordinator 决定是否 re-plan
- **Coordinator 崩溃**：Worker 保持在线等重连；GUI 显示 "连接丢失" 状态
- **LLM 调用失败**：Mini-Agent 内部重试机制（已有），失败后 Worker 上报 TASK FAILED

## 7. 测试策略

- **单元测试**：Tool 函数独立测试（不依赖 Mini-Agent，直接调用 → 验证 ROS service 调用）
- **集成测试**：A2A task 发送 → Worker 接收 → Tool 执行（Mock ROS services）
- **端到端测试**：GUI → Coordinator → Worker → ROS service（全链路，需要 Gazebo 或 mock）
- **回归测试**：现有 ROS launch 流程是否正常工作，保留的核心功能无退化

## 8. 实施顺序

1. **Phase 1**：Worker 改造一个 robot（如 cleaning_bot），验证 Tool Calling 模式
2. **Phase 2**：迁移其余 3 个 robot Worker
3. **Phase 3**：Coordinator 配置 + RouterAgent 增强（agents.yaml + 任务分解逻辑）
4. **Phase 4**：GUI HTTP 集成 + 去掉 task_manager.py
5. **Phase 5**：清理 4 个 robot_llm.py 中的 LLM 调用死代码
