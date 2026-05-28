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
