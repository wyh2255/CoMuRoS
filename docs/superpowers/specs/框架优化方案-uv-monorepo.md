# CoMuRoS + my_a2a 框架优化方案：uv workspace monorepo

> 基于《框架.md》当前架构分析，解决 CoMuRoS 和 my_a2a 两个模块互相调用、路径耦合、依赖管理两套体系的问题。
>
> **目标**：新机器上 `git clone --recurse-submodules + uv sync + colcon build` 一键完成安装。

---

## 一、当前问题诊断

### 问题 1：sys.path 运行时注入（×4 处）

两端共 4 处代码在进程启动时用相对路径硬编码把对方代码塞进 Python 路径：

| # | 位置 | 文件 | 具体操作 |
|---|------|------|----------|
| 1 | CoMuRoS → my_a2a | `robot_llm/robot_llm/a2a_worker_node.py:22-39` | `_resolve_a2a_paths()` 用 `../../../../my_a2a/src` 算路径 → `sys.path.insert` |
| 2 | my_a2a → Mini-Agent | `coordinator/router.py:147` | `sys.path.insert(0, mini_agent_path)` 然后 `from mini_agent.llm import LLMClient` |
| 3 | my_a2a → Mini-Agent | `worker/mini_agent_adapter.py:38-40` | `sys.path.insert(0, mini_agent_path)` 然后 `from mini_agent.agent import Agent` |
| 4 | my_a2a → CoMuRoS | `tests/integration/conftest.py:6-16` | `../../../../CoMuRoS/CoMuRoS` 算路径 → `sys.path.insert` 四次 |

**问题**：依赖目录布局硬编码为兄弟关系；无法 pip install；无法版本管理；换电脑必须手工对齐目录。四人分别维护自己的路径解析逻辑，一旦目录结构变化需要同时改 4 处。

### 问题 2：Mini-Agent 三层 vendor 嵌套

`my_a2a/src/Mini-Agent/` 包含 1094 个文件的三层嵌套复制：

```
Mini-Agent/                       # 第一层
├── mini_agent/tools/base.py      # ← 实际使用的导入路径
├── src/
│   └── Mini-Agent/               # 第二层（完全复制）
│       ├── mini_agent/...
│       └── src/
│           └── Mini-Agent/       # 第三层（完全复制）
│               └── mini_agent/...
```

经检查：**没有本地修改**（`git diff` 空），但`pyproject.toml` 同时又声明了 `mini-agent @ git+...` 作为 optional dep——两个源存在同步风险。且 `router.py`、`mini_agent_adapter.py`、`a2a_worker_node.py` 三个文件各有一套 Mini-Agent 路径发现逻辑。

### 问题 3：依赖管理两套体系

| 维度 | CoMuRoS | my_a2a |
|------|---------|--------|
| 管理工具 | pip + `requirements.txt` | uv + `pyproject.toml` |
| 锁文件 | 无 | `uv.lock` |
| ROS 2 构建 | `colcon build` + `setup.py` | N/A |
| 交叉依赖 | 运行时 sys.path hack | 测试时 sys.path hack |
| 公共依赖（如 httpx） | `requirements.txt` 未列 | 已在 pyproject.toml |

### 问题 4：配置副本

`my_a2a/config/agents.yaml` 描述的是 CoMuRoS 的机器人（cleaning-bot、delivery-bot 等）。CoMuRoS 增减机器人时，两处需要同步修改。

### 问题 5：新机器复现步骤多

1. git clone CoMuRoS（一个仓库）
2. git clone my_a2a（另一个仓库），且需保证是 `../my_a2a` 兄弟关系
3. `pip install -r CoMuRoS/requirements.txt`
4. `cd CoMuRoS && colcon build`
5. `cd my_a2a && uv sync`（可能与 CoMuRoS 的依赖有版本冲突）
6. 可能需要手动设置 `MY_A2A_PATH` 环境变量

---

## 二、解决方案

### 核心思路

```
现状（双向耦合）:
  CoMuRoS ──运行时 sys.path hack──→ my_a2a
  my_a2a  ──测试时 sys.path hack──→ CoMuRoS

优化后（单向依赖）:
  openharness-a2a ──pip dep──→ comuros-core ←──colcon dep──→ ROS 2 robot packages
```

**不在已有仓库之外新建容器仓库**，而是直接在 CoMuRoS 仓库内做三件事：

1. **加 `pyproject.toml`**：uv 管理所有 Python 依赖
2. **加 git submodule**：`my_a2a` 作为子模块纳入统一管理
3. **删 4 处 `sys.path.insert`**：改为标准 pip import

### 最终目录结构

```
CoMuRoS/                                    ← 已有 git 仓库
├── pyproject.toml                          ← ● NEW: uv workspace root
├── uv.lock                                 ← ● NEW: 统一锁文件
├── .gitmodules                             ← ● NEW: 声明 my_a2a submodule
│
├── my_a2a/                                 ← ● NEW: git submodule
│   ├── pyproject.toml                      ← 已有，小改（mini-agent 升为必选 dep）
│   ├── src/openharness_a2a/
│   │   ├── coordinator/router.py           ← ● CHANGED: 删 sys.path.insert
│   │   └── worker/mini_agent_adapter.py    ← ● CHANGED: 删 sys.path.insert
│   └── tests/integration/conftest.py       ← ● CHANGED: 删 sys.path.insert
│
├── CoMuRoS/                                ← colcon 工作区（ROS 2 包）
│   ├── chatty/
│   ├── cleaning_bot/
│   ├── delivery_bot/
│   ├── drone/
│   ├── robot_llm/
│   │   └── robot_llm/
│   │       └── a2a_worker_node.py         ← ● CHANGED: 删 _resolve_a2a_paths()
│   ├── robot_interface/
│   └── robots/
│
├── config/
│   └── agents.yaml                        ← ● NEW: 统一配置源
│
├── src/
│   └── comuros_core/                      ← ● NEW（可选 Phase）：公共代码包
│       └── __init__.py
│
├── docs/
│   └── superpowers/specs/
│       ├── 框架.md                         ← 已有
│       └── 框架优化方案-uv-monorepo.md      ← ● NEW: 本文档
│
└── requirements.txt                       ← 可删除（由 pyproject.toml 替代）
```

---

## 三、依赖关系详细设计

### pyproject.toml

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "comuros"
version = "0.1.0"
requires-python = ">=3.10"
description = "CoMuRoS - Collaborative Multi-Robot System"

dependencies = [
    "openharness-a2a",
    "mini-agent @ git+https://github.com/MiniMax-AI/Mini-Agent.git",
    # GUI
    "customtkinter>=5.2.0",
    "httpx>=0.27.0",
    # LLM
    "openai>=1.57.0",
    "google-generativeai>=0.8.0",
    "ollama>=0.4.0",
    "requests>=2.31.0",
    # Speech / Audio
    "sounddevice>=0.4.6",
    "scipy>=1.10.0",
    "numpy>=1.23.0",
    "pydub>=0.25.1",
    "edge-tts>=6.1.10",
    "torch>=2.0.0",
    "openai-whisper>=20231117",
    "regex>=2023.10.3",
    # Utilities
    "pydantic>=2.0.0",
]

[tool.uv.sources]
openharness-a2a = { path = "my_a2a", editable = true }

[tool.uv.workspace]
members = ["my_a2a"]

[tool.uv]
dev-dependencies = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.23.0",
]
```

### my_a2a/pyproject.toml 的修改

```toml
# 将 mini-agent 从 optional 升为必选
dependencies = [
    # ... 原有依赖不变 ...
    "mini-agent @ git+https://github.com/MiniMax-AI/Mini-Agent.git",  # ← 新增此行
]

# 删除 [project.optional-dependencies] mini-agent 条目
# 删除 src/Mini-Agent/ 整个目录
```

### uv workspace 工作原理

```
uv sync (在 CoMuRoS/ 根目录执行)
  ├── 安装 comuros (当前 project)
  │     ├── customtkinter, openai, httpx, ...
  │     ├── openharness-a2a  (path = my_a2a/, editable 模式)
  │     │     └── fastapi, uvicorn, a2a-sdk, ...
  │     └── mini-agent  (git source)
  └── 统一写入 uv.lock，锁定所有依赖版本
```

- `uv sync` 在根目录创建 `.venv/`
- `openharness-a2a` 以 editable 模式安装，修改 my_a2a 源码即时生效（无需重新 uv sync）
- `mini-agent` 从 git 下载安装，没有版本漂移

### uv + colcon 共存

```
# 运行时 PYTHONPATH 解析链
source .venv/bin/activate               # 将 .venv/lib/python3.x/site-packages 加入路径
source CoMuRoS/install/setup.bash       # 将 colcon install/ 加入路径

# 导入解析顺序：
#   1. colcon install/ 里的 ament 包（cleaning_bot, drone, chatty 等 ROS 包）
#   2. .venv/site-packages 里的 uv 包（openharness-a2a, mini-agent, httpx 等）
# 两者不冲突，因为 ament 包和 pip 包的命名空间不重叠
```

---

## 四、实施步骤

### Phase 1：引入 uv + 消灭 sys.path hack（1-2 小时）

#### Step 1.1：创建 pyproject.toml

在 CoMuRoS 仓库根目录创建 `pyproject.toml`，内容见第三章。

#### Step 1.2：添加 my_a2a submodule

```bash
cd CoMuRoS/
git submodule add <my_a2a-git-url> my_a2a
```

如果 my_a2a 暂时没有独立的远程仓库，或在同一台机器上开发，也可以先用本地路径：

```toml
[tool.uv.sources]
openharness-a2a = { path = "../my_a2a", editable = true }
```

等到需要多机器复现时再切到 submodule。

#### Step 1.3：删除 a2a_worker_node.py 的路径 hack

删除 `CoMuRoS/server/robot_llm/robot_llm/a2a_worker_node.py` 的 L22-39：

```python
# ===== 删除开始 =====
def _resolve_a2a_paths():
    """Resolve my_a2a and Mini-Agent paths from env vars or fallback to relative paths."""
    a2a_path = os.environ.get("MY_A2A_PATH")
    mini_agent_path = os.environ.get("MINI_AGENT_PATH")
    if not a2a_path:
        a2a_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "my_a2a", "src")
    if not mini_agent_path:
        mini_agent_path = os.path.join(a2a_path, "Mini-Agent")
    return os.path.abspath(a2a_path), os.path.abspath(mini_agent_path)

_A2A_PATH, _MINI_AGENT_PATH = _resolve_a2a_paths()
if _A2A_PATH not in sys.path:
    sys.path.insert(0, _A2A_PATH)
if _MINI_AGENT_PATH not in sys.path:
    sys.path.insert(0, _MINI_AGENT_PATH)
# ===== 删除结束 =====
```

如果将来需要考虑向后兼容，可以保留 `MY_A2A_PATH` 和 `MINI_AGENT_PATH` 环境变量的读取，只打印 deprecation warning 而不做 `sys.path.insert`：

```python
import warnings
if os.environ.get("MY_A2A_PATH"):
    warnings.warn("MY_A2A_PATH is no longer needed. Remove it from your environment.", DeprecationWarning)
```

#### Step 1.4：修改 my_a2a/router.py

```python
# 删除 _get_mini_agent_path() 函数及模块级 _MINI_AGENT_PATH

class RouterAgent:
    def __init__(self, ...):
        # self._mini_agent_path = _MINI_AGENT_PATH  # ← 删除
        ...

    def _route_with_llm(self, ...):
        # sys.path.insert(0, str(self._mini_agent_path))  # ← 删除
        from mini_agent.llm import LLMClient  # ← 直接 import，由 site-packages 提供
        ...
```

#### Step 1.5：修改 my_a2a/mini_agent_adapter.py

同理删除 `_get_mini_agent_path()`、模块级变量和 `sys.path.insert`。

#### Step 1.6：修改 my_a2a/conftest.py

删除所有 `sys.path.insert`，只保留 pytest 配置：

```python
"""pytest configuration for integration tests.

All Python dependencies (openharness-a2a, mini-agent, comuros-core) 
are installed via 'uv sync' from the workspace root.
ROS 2 packages (cleaning_bot, drone, etc.) are provided by 
sourcing 'CoMuRoS/install/setup.bash' after colcon build.
"""
import pytest

def pytest_configure(config):
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
```

#### Step 1.7：删除 vendored Mini-Agent + 升级依赖

```bash
cd my_a2a
git rm -r src/Mini-Agent/
```

修改 `my_a2a/pyproject.toml`，将 `mini-agent` 从 optional 升为必选：

```toml
dependencies = [
    # ... 原有 ...
    "mini-agent @ git+https://github.com/MiniMax-AI/Mini-Agent.git",  # ← 新增
]
# 同时删除 [project.optional-dependencies] 中的 mini-agent 条目
```

#### Step 1.8：验证

```bash
cd CoMuRoS/

# uv 安装
uv sync

# 验证 import 能工作
uv run python -c "from openharness_a2a.worker.a2a_server import create_worker_a2a_server; print('openharness-a2a OK')"
uv run python -c "from mini_agent.tools.base import Tool; print('mini-agent OK')"
uv run python -c "from robot_llm.a2a_worker_node import A2AWorkerNode; print('a2a_worker_node OK')"

# colcon 构建
source .venv/bin/activate
cd CoMuRoS/
colcon build --symlink-install
cd ..

# 运行集成测试
source .venv/bin/activate
source CoMuRoS/install/setup.bash
uv run pytest my_a2a/tests/
```

---

### Phase 2：配置集中化（可选，建议同步做）

#### Step 2.1：统一 agents.yaml

把 `my_a2a/config/agents.yaml` 移到 `CoMuRoS/config/agents.yaml`。

修改 `agent_registry.py`：

```python
class AgentRegistry:
    def __init__(self, config_path: str | None = None):
        if config_path is None:
            config_path = os.environ.get(
                "AGENTS_CONFIG_PATH",
                os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "agents.yaml"),
            )
        ...
```

在启动脚本中传递：

```bash
openharness-a2a --config-path config/agents.yaml
```

#### Step 2.2：删除环境变量引用

在所有代码和 launch 文件中删除 `MY_A2A_PATH` 和 `MINI_AGENT_PATH` 的引用。

```bash
grep -rn "MY_A2A_PATH\|MINI_AGENT_PATH" . --include="*.py" --include="*.launch.py" --include="*.yaml" --include="*.md"
```

---

### Phase 3：公共代码提取（可选，按需做）

如果 robot 包中出现重复代码，可以提取到 `src/comuros_core/`。

**前提**：至少在两个 robot 包中出现了重复的工具逻辑或辅助方法。

示例：提取共同的基础 Tool traits 或 step callback 处理。

```bash
mkdir -p src/comuros_core
touch src/comuros_core/__init__.py
```

在 `pyproject.toml` 中添加：

```toml
[tool.hatch.build.targets.wheel]
packages = ["src/comuros_core"]
```

其他 robot 包可以 `from comuros_core import ...`。

---

## 五：在新机器上复现

### 前置条件

```bash
# 1. ROS 2 Humble（或其他 distro）
sudo apt install ros-humble-desktop python3-colcon-common-extensions

# 2. uv
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 安装

```bash
# 克隆（带 submodule）
git clone --recurse-submodules <repo-url>
cd CoMuRoS

# uv 安装所有 Python 依赖
uv sync

# colcon 构建
source /opt/ros/humble/setup.bash
cd CoMuRoS
colcon build --symlink-install
cd ..

echo "安装完成"
```

### 运行

```bash
# 所有终端都要执行
source .venv/bin/activate
source CoMuRoS/install/setup.bash

# Terminal 1: 仿真
ros2 launch multi_robot multi_robot.launch.py

# Terminal 2: 机器人类人接口
ros2 launch robot_llm robot_llms.launch.py
ros2 launch robot_llm robot_services.launch.py

# Terminal 3: GUI
ros2 launch chatty chat_system.launch.py

# Terminal 4: Coordinator
openharness-a2a
```

### 与当前流程对比

| 步骤 | 优化前 | 优化后 |
|------|--------|--------|
| 克隆 | `git clone repo1 && git clone repo2`（2 次） | `git clone --recurse-submodules`（1 次） |
| 目录对齐 | 必须兄弟关系：`../my_a2a` | 不需要，submodule 自动在正确位置 |
| Python 依赖 | `pip install -r requirements.txt` + `cd my_a2a && uv sync` | `uv sync`（一个命令，统一 lockfile） |
| COS 2 构建 | `colcon build` | `colcon build`（不变） |
| env 变量 | 可能需要设 `MY_A2A_PATH` | 不需要设置任何环境变量 |
| 依赖版本冲突 | pip 和 uv 各管各的，可能冲突 | uv.lock 统一锁定，无冲突 |

---

## 附录 A：文件变更汇总

### 新增文件

| 文件 | 说明 |
|------|------|
| `CoMuRoS/pyproject.toml` | uv workspace root，依赖声明 |
| `CoMuRoS/uv.lock` | uv 锁文件 |
| `CoMuRoS/.gitmodules` | my_a2a submodule 声明 |
| `CoMuRoS/config/agents.yaml` | 统一 Agent 配置 |

### 修改文件

| 文件 | 修改内容 |
|------|----------|
| **CoMuRoS 侧** | |
| `CoMuRoS/CoMuRoS/robot_llm/robot_llm/a2a_worker_node.py` | 删除 `_resolve_a2a_paths()` + `sys.path.insert` |
| **my_a2a 侧** | |
| `my_a2a/pyproject.toml` | `mini-agent` 从 optional 升为必选 dep |
| `my_a2a/src/openharness_a2a/coordinator/router.py` | 删除 `sys.path.insert` + `_get_mini_agent_path()` |
| `my_a2a/src/openharness_a2a/worker/mini_agent_adapter.py` | 删除 `sys.path.insert` + `_get_mini_agent_path()` |
| `my_a2a/tests/integration/conftest.py` | 删除所有 `sys.path.insert` |

### 删除文件

| 文件 | 说明 |
|------|------|
| `my_a2a/src/Mini-Agent/` | 1094 文件的 vendored 副本 → 改为 git dep |
| `my_a2a/config/agents.yaml` | 移到 `CoMuRoS/config/agents.yaml` |
| `CoMuRoS/requirements.txt` | 可选删除，由 `pyproject.toml` 替代 |

---

## 附录 B：风险与回退

| 风险 | 概率 | 缓解措施 |
|------|------|---------|
| Mini-Agent git dep 和当前 vendored 版本行为不一致 | 低 | 已验证 vendored 副本无本地修改。在 `pyproject.toml` 中 pin commit SHA：`mini-agent @ git+https://github.com/MiniMax-AI/Mini-Agent.git@<sha>` |
| uv + colcon 的 PYTHONPATH 冲突 | 极低 | `source .venv/bin/activate` 在 `source install/setup.bash` 之前即可。两者导入路径不重叠 |
| submodule 管理增加复杂度 | 中 | 如果团队不熟悉 submodule，退路：不用 submodule，用 `[tool.uv.sources]` 的 `path` 指向本地路径，安装时手动 clone my_a2a 到期望位置 |
| 团队成员未装 uv | 低 | `setup.sh` 内自动检测并安装：`which uv || curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| Phase 2 中 `agent_registry.py` 默认路径变了 | 中 | 保持向后兼容：`config_path` 参数默认 `None` 时先读 `AGENTS_CONFIG_PATH` env，再 fallback 到新默认路径 |

---

## 附录 C：验证 checklist

Phase 1 完成后，逐项验证：

```bash
# [ ] uv sync 成功，无依赖冲突
uv sync

# [ ] openharness-a2a 可导入
uv run python -c "from openharness_a2a.coordinator.server import app; print('OK')"

# [ ] mini-agent 可导入（来自 git，不再来自 vendored 副本）
uv run python -c "from mini_agent.agent import Agent; print('OK')"

# [ ] a2a_worker_node 可导入（不再需要 sys.path hack）
source .venv/bin/activate
cd CoMuRoS && colcon build --symlink-install
uv run python -c "from robot_llm.a2a_worker_node import A2AWorkerNode; print('OK')"

# [ ] my_a2a 集成测试通过
source install/setup.bash
cd ..
uv run pytest my_a2a/tests/integration/ -v

# [ ] 旧的 sys.path 代码已全部删除
! grep -rn "sys.path.insert.*a2a\|_resolve_a2a_paths\|_get_mini_agent_path" --include="*.py"

# [ ] 无 vendored Mini-Agent 残留
test ! -d my_a2a/src/Mini-Agent && echo "已清理"
```

---

## 附录 D：为什么不推荐更复杂的改动

| 被否方案 | 理由 |
|----------|------|
| 创建全新的 comuros-platform 容器仓库 | 多一层仓库嵌套，git 操作复杂（subtree / submodule 交叉），收益没有更高 |
| 把 A2AWorkerNode 提取为独立 pip 包 | 它继承 `rclpy.node.Node`，强依赖 ROS，放到独立包中只是改变了安装方式，没有减少耦合，反而增加了维护负担 |
| 完全用 uv 替代 colcon | ROS 2 的 ament/colcon 体系还处理 launch 文件、package.xml、消息/服务生成等，uv 无法替代 |
| 把 my_a2a 代码直接合并进 CoMuRoS | 两个模块的生命周期不同（my_a2a 偏基础设施，CoMuRoS 偏机器人应用），合并后界限模糊，不利于各自独立演进 |
