#!/usr/bin/env python3
"""Cleaning bot A2A Worker node."""

import time
import subprocess

import rclpy
from rclpy.executors import MultiThreadedExecutor
from robot_interface.srv import GotoPoseHolonomic
from std_msgs.msg import Bool

from robot_llm.a2a_worker_node import A2AWorkerNode, TaskCancelledException
from cleaning_bot.tools import CleanTool


ROBOT_NAME = "cleaning_bot"
ROBOT_TYPE = "Holonomic Drive Robot"
PACKAGE_NAME = "cleaning_bot"
NODE_NAME = "cleaning_bot_llm_node"
A2A_PORT = 8091


class CleaningBotWorker(A2AWorkerNode):
    """Cleaning bot: A2A Worker with CleanTool."""

    def __init__(self):
        super().__init__(
            node_name=NODE_NAME,
            robot_name=ROBOT_NAME,
            package_name=PACKAGE_NAME,
            port=A2A_PORT,
        )

        # GoTo service client
        self._goto_client = self.create_client(
            GotoPoseHolonomic, "/r1/goto_pose", callback_group=self.multi_group
        )
        self._cancel_goto_pub = self.create_publisher(Bool, "/r1/cancel_goto_pose_goal", 10)

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
