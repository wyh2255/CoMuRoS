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
