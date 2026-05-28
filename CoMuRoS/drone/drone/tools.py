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
            success = self._node.query_callback(prompt)
            if success:
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
