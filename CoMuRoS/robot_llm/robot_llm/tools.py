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
