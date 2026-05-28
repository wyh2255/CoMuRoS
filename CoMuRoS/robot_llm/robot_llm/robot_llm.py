#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Robot arm A2A Worker node."""

import time

import rclpy
from rclpy.executors import MultiThreadedExecutor
from robot_interface.srv import StartPick
from std_msgs.msg import String

from robot_llm.a2a_worker_node import A2AWorkerNode, TaskCancelledException
from robot_llm.tools import PickObjectTool


ROBOT_NAME = "robot1"
ROBOT_TYPE = "Robotic Arm"
PACKAGE_NAME = "robot_llm"
NODE_NAME = "robot_llm_node"
A2A_PORT = 8094


class RobotArmWorker(A2AWorkerNode):
    """Robot arm: A2A Worker with PickObjectTool."""

    def __init__(self):
        super().__init__(
            node_name=NODE_NAME,
            robot_name=ROBOT_NAME,
            package_name=PACKAGE_NAME,
            port=A2A_PORT,
        )

        self._pick_client = self.create_client(StartPick, '/start_pick', callback_group=self.multi_group)
        self._cancel_pub = self.create_publisher(String, '/start_pick/cancel', 10)

        self.start_a2a_server()

    def _get_tools(self) -> list:
        return [PickObjectTool(node=self)]

    def _get_capabilities(self) -> list[str]:
        return ["pick_and_place"]

    def _get_system_prompt(self) -> str:
        return (
            f"You control a {ROBOT_TYPE} named '{ROBOT_NAME}'. "
            "Use the 'pick_object' tool to pick objects by color. "
            "Valid objects: 'green object', 'brown object', 'grey object'. "
            "Report task status updates when completing each action."
        )

    def pick_object(self, object_name: str = "red_gear") -> bool:
        """Call StartPick service to pick the specified object."""
        self.check_cancelled()
        self.get_logger().info(f"Picking '{object_name}'...")

        if not self._pick_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("StartPick service not available!")
            return False

        req = StartPick.Request()
        req.object_name = object_name

        future = self._pick_client.call_async(req)
        self.get_logger().info(f"Waiting for /start_pick response for '{object_name}'...")

        deadline = time.time() + 1200000.0
        while rclpy.ok() and not future.done():
            self.check_cancelled()
            time.sleep(0.05)
            if time.time() > deadline:
                self.get_logger().error(f"Service call for '{object_name}' timed out.")
                return False

        try:
            res = future.result()
        except Exception as e:
            self.get_logger().error(f"Service call failed: {e}")
            return False

        if res.success:
            self.get_logger().info(f"Pick succeeded: {res.message}")
            return True
        else:
            self.get_logger().warn(f"Pick failed: {res.message}")
            return False

    def pick_green_object(self) -> bool:
        """Pick the green object."""
        self.get_logger().info("Picking green object...")
        success = self.pick_object("green object")
        if success:
            self.robot_task_completed("pick green object")
        else:
            self.robot_task_interrupted("pick green object")
        return success

    def pick_brown_object(self) -> bool:
        """Pick the brown object."""
        self.get_logger().info("Picking brown object...")
        success = self.pick_object("brown object")
        if success:
            self.robot_task_completed("pick brown object")
        else:
            self.robot_task_interrupted("pick brown object")
        return success

    def pick_grey_object(self) -> bool:
        """Pick the grey object."""
        self.get_logger().info("Picking grey object...")
        success = self.pick_object("grey object")
        if success:
            self.robot_task_completed("pick grey object")
        else:
            self.robot_task_interrupted("pick grey object")
        return success


def main(args=None):
    rclpy.init(args=args)
    node = RobotArmWorker()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
