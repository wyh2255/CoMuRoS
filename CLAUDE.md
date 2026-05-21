# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CoMuRoS (Collaborative Multi-Robot System) is a ROS 2 Python system enabling natural-language interaction with heterogeneous robot teams using LLMs (OpenAI, Gemini, Ollama, XAI). It performs hierarchical task planning with event-driven replanning.

## Build Commands

```bash
# Build all packages
colcon build --symlink-install
source install/setup.bash

# Build a single package
colcon build --packages-select <package_name>
source install/setup.bash

# Run tests (ament)
colcon test
colcon test-result --verbose
```

## Run the System (4 terminals)

```bash
# Terminal 1: Simulation environment
ros2 launch multi_robot multi_robot.launch.py

# Terminal 2: Robot LLM interfaces
ros2 launch robot_llm robot_llms.launch.py

# Terminal 3: Robot control services
ros2 launch robot_llm robot_services.launch.py

# Terminal 4: Chat interface
ros2 launch chatty chat_system.launch.py

# Optional model/config for chat
ros2 launch chatty chat_system.launch.py model:=1 config_file:=robot_config_roscon_2025
ros2 launch chatty chat_system.launch.py enable_audio_input:=true enable_audio_output:=true
```

## Architecture

### ROS 2 Package Layout

```
CoMuRoS/
├── chatty/              # Central chat system (Python package)
│   ├── chatty/
│   │   ├── chat_gui.py          # customtkinter GUI for user interaction
│   │   ├── chat_manager.py       # Message routing, conversation history
│   │   ├── task_manager.py       # LLM-based task planner (OpenAI/Gemini/Ollama/XAI)
│   │   ├── speak.py              # Text-to-speech (edge-tts)
│   │   ├── listen.py             # Speech-to-text (whisper)
│   │   └── microphone.py         # Audio capture
│   ├── config/robot_config_*.json  # Robot capability/morphology definitions
│   ├── launch/chat_system.launch.py
│   └── setup.py
│
├── cleaning_bot/        # Cleaning robot package
│   ├── cleaning_bot_llm.py       # LLM interface for cleaning tasks
│   └── holonomic_position_controller_service.py
│
├── delivery_bot/        # Delivery robot package
│   └── delivery_bot_llm.py
│
├── drone/               # Drone package
│   └── drone_llm.py + position controller services
│
├── robot_llm/           # Meta-package with launch files for all robots
│   ├── launch/robot_llms.launch.py     # Launches all robot LLM nodes
│   ├── launch/robot_services.launch.py # Launches all position controllers
│   └── robot_llm/robot_llm.py          # Generic LLM node for robot arm
│
├── robot_interface/     # Custom ROS 2 actions (PickObject) and services (GoTo, Find, StartPick)
│
└── robots/
    ├── multi_robot/           # Ignition Gazebo food court simulation
    ├── x3_uav/                # X3 UAV drone (description + ignition + LLM)
    └── yahboom/               # Yahboom Rosmaster X3 ground robots
```

### Data Flow

```
User Input (GUI text or microphone)
  → /chat/input topic → ChatManager (stores history, routes messages)
  → /chat/output topic → TaskManager (LLM parses natural language → structured tasks)
  → /task_manager/tasks_json topic → Robot LLM nodes (subscribe per-robot)
  → Position controller services (holonomic, drone) via custom ROS 2 services
  → Gazebo simulation
```

### Multi-LLM Support

The `task_manager.py` supports switching between OpenAI, Gemini, XAI, and Ollama. Model selection is configured via the `model` launch argument (numeric index mapping at top of file). API keys are read from environment variables (`OPENAI_API_KEY`, `GEMINI_API_KEY`, `XAI_API_KEY`).

### Adding a New Robot

1. Add to `multi_robot.launch.py` (Gazebo spawn)
2. Create robot package with LLM node + position controller service
3. Add LLM node to `robot_llms.launch.py`
4. Add controller to `robot_services.launch.py`
5. Update config JSON in `chatty/config/`

## Key Details

- **API keys**: Set `OPENAI_API_KEY`, `GEMINI_API_KEY`, `XAI_API_KEY` in environment
- **System dependencies**: `ffmpeg`, `portaudio19-dev` (for audio)
- **GUI**: customtkinter-based chat interface with colored per-robot message styling
- **Event-driven replanning**: TaskManager re-plans for all robots when it receives "Event:" lines on `/chat/output`
- **Simulation**: Ignition Gazebo with food court world, 3 robot types spawned with staggered delays
- **Namespace convention**: r1 (cleaning), r2 (delivery), r3 (drone)
