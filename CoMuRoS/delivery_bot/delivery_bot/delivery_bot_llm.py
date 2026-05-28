#!/usr/bin/env python3
"""Delivery bot A2A Worker node."""

import time
import subprocess

import rclpy
from rclpy.executors import MultiThreadedExecutor
from robot_interface.srv import GotoPoseHolonomic
from std_msgs.msg import Bool

from robot_llm.a2a_worker_node import A2AWorkerNode, TaskCancelledException
from delivery_bot.tools import DeliverFoodTool, ClearTableTool


ROBOT_NAME = "delivery_bot"
ROBOT_TYPE = "Differential Drive Robot"
PACKAGE_NAME = "delivery_bot"
NODE_NAME = "delivery_bot_llm_node"
A2A_PORT = 8092

# Location constants
TableLocation = {
    1: [0.0, -1.3, 0.0],
    2: [3.0, -1.3, 0.0],
    3: [6.0, -1.3, 0.0],
    4: [9.0, -1.3, 0.0],
}

StallLocation = {
    1: [0.0, 0.0, 0.0],
    2: [4.0, 0.0, 0.0],
    3: [8.0, 0.0, 0.0],
}

home_pose = [0.0, 0.0, 0.0]
sink_pose = [-1.85, -0.5, 0.0]
food = ["food1", "food2", "food3"]


class DeliveryBotWorker(A2AWorkerNode):
    """Delivery bot: A2A Worker with DeliverFoodTool and ClearTableTool."""

    def __init__(self):
        super().__init__(
            node_name=NODE_NAME,
            robot_name=ROBOT_NAME,
            package_name=PACKAGE_NAME,
            port=A2A_PORT,
        )

        # GoTo service client (r2 namespace for delivery bot)
        self._goto_client = self.create_client(
            GotoPoseHolonomic, "/r2/goto_pose", callback_group=self.multi_group
        )
        self._cancel_goto_pub = self.create_publisher(Bool, "/r2/cancel_goto_pose_goal", 10)

        self.start_a2a_server()

    def _get_tools(self) -> list:
        return [DeliverFoodTool(node=self), ClearTableTool(node=self)]

    def _get_capabilities(self) -> list[str]:
        return ["food_delivery", "table_clearing"]

    def _get_system_prompt(self) -> str:
        return (
            f"You control a {ROBOT_TYPE} named '{ROBOT_NAME}' in a restaurant food court. "
            "You can deliver food from stalls to tables and clear dishes from tables to the sink. "
            "Use the 'deliver_food' tool to deliver food, and 'clear_table' to clear dishes. "
            "Report task status updates when completing each action."
        )

    def goto_service(self, x: float, y: float, yaw_deg: float) -> bool:
        self.check_cancelled()
        if not self._goto_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("GotoPoseHolonomic service NOT available!")
            return False
        req = GotoPoseHolonomic.Request()
        req.x = x
        req.y = y
        req.yaw_deg = yaw_deg
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

    def teleport(self, name, x, y, z):
        self.check_cancelled()
        self.get_logger().info(f"Teleport Object: {name}")
        req = (
            f'name: "{name}", '
            f"position: {{x: {x}, y: {y}, z: {z}}}, "
            "orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}"
        )
        cmd = [
            "ign", "service", "-s", "/world/food_court/set_pose",
            "--reqtype", "ignition.msgs.Pose",
            "--reptype", "ignition.msgs.Boolean",
            "--timeout", "1000",
            "--req", req,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        self.get_logger().info(f"Teleport result: {result.stdout}")
        return result.returncode == 0

    def deliver_food(self, stall_number, table_number):
        """Deliver food from stall to table."""
        self.get_logger().info("Delivering food ...")

        self.update_robot_state({
            "current_task": "deliver_food",
            "task_status": "in_progress",
            "stall_number": stall_number,
            "table_number": table_number,
            "delivery_stage": "started",
        })

        table_pose = TableLocation.get(table_number)
        stall_pose = StallLocation.get(stall_number)

        if table_pose is None:
            self.update_robot_state({
                "task_status": "failed",
                "failure_reason": f"invalid_table_number_{table_number}",
            })
            raise ValueError(f"Invalid table number: {table_number}")

        if stall_pose is None:
            self.update_robot_state({
                "task_status": "failed",
                "failure_reason": f"invalid_stall_number_{stall_number}",
            })
            raise ValueError(f"Invalid stall number: {stall_number}")

        sx, sy, syaw = stall_pose
        hx, hy, hyaw = table_pose
        home_pose_x, home_pose_y, home_pose_yaw = home_pose
        food_name = food[stall_number - 1]

        self.check_cancelled()

        # Navigate to stall
        self.update_robot_state({
            "task_status": "navigating_to_stall",
            "delivery_stage": "going_to_stall",
            "target_coords": {"x": sx, "y": sy + 0.5, "yaw": syaw},
            "food_item": food_name,
        })
        self.goto_service(sx, sy + 0.5, syaw)
        self.get_logger().info("Robot Near Stall")
        time.sleep(2.0)

        # Pick food
        self.update_robot_state({
            "task_status": "picking_food",
            "delivery_stage": "at_stall",
            "current_location": f"stall_{stall_number}",
        })
        self.teleport(name=food_name, x=sx, y=sy + 0.5, z=0.4)
        self.get_logger().info("Food Picked from Stall")
        time.sleep(3.0)

        # Navigate to table
        self.update_robot_state({
            "task_status": "navigating_to_table",
            "delivery_stage": "carrying_food",
            "food_picked": True,
            "target_coords": {"x": hx, "y": hy, "yaw": hyaw},
        })
        self.goto_service(hx, hy, hyaw)
        self.get_logger().info("Robot Near Table")
        time.sleep(2.0)

        # Deliver food to table
        self.update_robot_state({
            "task_status": "delivering_food",
            "delivery_stage": "at_table",
            "current_location": f"table_{table_number}",
        })
        self.teleport(name=food_name, x=hx + 0.1, y=hy - 0.7, z=0.6)
        self.get_logger().info("Food Delivered to table")
        time.sleep(3.0)

        # Return home
        self.update_robot_state({
            "task_status": "returning_home",
            "delivery_stage": "going_home",
            "food_delivered": True,
            "delivery_completed_at": time.time(),
        })
        self.goto_service(home_pose_x, home_pose_y, home_pose_yaw)
        self.get_logger().info("Robot Went Home")

        # Complete
        self.update_robot_state({
            "task_status": "completed",
            "delivery_stage": "at_home",
            "current_location": "home",
            f"delivery_stall{stall_number}_to_table{table_number}": "completed",
            "completion_timestamp": time.time(),
        })

    def clear_table(self, table_number, food_name):
        """Clear dishes from table to sink."""
        self.get_logger().info("Clearing Table ...")

        self.update_robot_state({
            "current_task": "clear_table",
            "task_status": "in_progress",
            "table_number": table_number,
            "food_to_clear": food_name,
            "clearing_stage": "started",
        })

        table_pose = TableLocation.get(table_number)
        self.get_logger().info(
            f"Clear the food item: {food_name} from table number: {table_number}"
        )

        if table_pose is None:
            self.update_robot_state({
                "task_status": "failed",
                "failure_reason": f"invalid_table_number_{table_number}",
            })
            raise ValueError(f"Invalid table number: {table_number}")

        table_x, table_y, table_yaw = table_pose
        sink_x, sink_y, sink_yaw = sink_pose
        home_pose_x, home_pose_y, home_pose_yaw = home_pose

        self.check_cancelled()

        # Navigate to table
        self.update_robot_state({
            "task_status": "navigating_to_table",
            "clearing_stage": "going_to_table",
            "target_coords": {"x": table_x, "y": table_y, "yaw": table_yaw},
        })
        self.goto_service(table_x, table_y, table_yaw)
        self.get_logger().info("Robot Near Table")
        time.sleep(2.0)

        # Pick food from table
        self.update_robot_state({
            "task_status": "picking_food_from_table",
            "clearing_stage": "at_table",
            "current_location": f"table_{table_number}",
        })
        self.teleport(name=food_name, x=table_x, y=table_y, z=0.4)
        self.get_logger().info("Food Picked from Table")
        time.sleep(3.0)

        # Navigate to sink
        self.update_robot_state({
            "task_status": "navigating_to_sink",
            "clearing_stage": "carrying_dishes",
            "food_picked": True,
            "target_coords": {"x": sink_x, "y": sink_y, "yaw": sink_yaw},
        })
        self.goto_service(sink_x, sink_y, sink_yaw)
        self.get_logger().info("Robot Near Sink")
        time.sleep(2.0)

        # Drop in sink
        self.update_robot_state({
            "task_status": "dropping_in_sink",
            "clearing_stage": "at_sink",
            "current_location": "sink",
        })
        self.teleport(name=food_name, x=-2.5, y=-0.5, z=0.6)
        self.get_logger().info("Food Dropped in Sink")
        time.sleep(3.0)

        # Return home
        self.update_robot_state({
            "task_status": "returning_home",
            "clearing_stage": "going_home",
            "dishes_cleared": True,
            "clearing_completed_at": time.time(),
        })
        self.goto_service(home_pose_x, home_pose_y, home_pose_yaw)
        self.get_logger().info("Robot Went Home")

        # Complete
        self.update_robot_state({
            "task_status": "completed",
            "clearing_stage": "at_home",
            "current_location": "home",
            f"table_{table_number}_cleared": True,
            "cleared_item": food_name,
            "completion_timestamp": time.time(),
        })


def main(args=None):
    rclpy.init(args=args)
    node = DeliveryBotWorker()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
