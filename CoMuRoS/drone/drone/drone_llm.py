#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Drone A2A Worker node.

This module implements the drone's core control node as an A2A Worker,
using Tools (HoverTool, DescribeSceneTool) managed by the Mini-Agent
framework instead of LLM code generation.
"""

import time
import base64

import rclpy
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String, Bool
from robot_interface.srv import GotoPoseDrone
from openai import OpenAI
import os

from robot_llm.a2a_worker_node import A2AWorkerNode
from drone.tools import HoverTool, DescribeSceneTool


# ==================== Drone configuration ====================
ROBOT_NAME = "drone"
ROBOT_TYPE = "Quadrotor UAV"
NODE_NAME = "drone_llm_node"
PACKAGE_NAME = "drone"
A2A_PORT = 8093

api_key = os.getenv("DEEPSEEK_API_KEY")


class DroneWorker(A2AWorkerNode):
    """Drone: A2A Worker with HoverTool and DescribeSceneTool.

    Capabilities:
      - aerial_inspection: 3D movement via hover/goto_service
      - scene_description: VLM-based scene analysis via bottom camera

    Key methods exposed to tools:
      - hover(): move to a 3D position with state tracking
      - query_callback(): send image + prompt to VLM and return answer
      - describe_screen(): orchestrate VLM query with full state management
    """

    def __init__(self):
        super().__init__(
            node_name=NODE_NAME,
            robot_name=ROBOT_NAME,
            package_name=PACKAGE_NAME,
            port=A2A_PORT,
        )

        # VLM (Vision Language Model) client and image cache
        self.Visionclient = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        self.latest_image_b64 = None

        # GoTo service client (drone-specific: x, y, z, yaw_deg)
        self._goto_client = self.create_client(
            GotoPoseDrone, "/r3/goto_pose", callback_group=self.multi_group
        )
        self._cancel_goto_pub = self.create_publisher(Bool, "/r3/cancel_goto_pose_goal", 10)
        self.pub_input_msg = self.create_publisher(String, "/chat/input", 10)

        # Subscribe to drone bottom camera (compressed image for VLM)
        self.create_subscription(
            CompressedImage,
            "/r3/bottom_camera/color/image_raw/compressed",
            self.image_callback, 10,
            callback_group=self.single_group,
        )

        self.start_a2a_server()

    # --- A2A Worker hooks ---

    def _get_tools(self) -> list:
        return [HoverTool(node=self), DescribeSceneTool(node=self)]

    def _get_capabilities(self) -> list[str]:
        return ["aerial_inspection", "scene_description"]

    def _get_system_prompt(self) -> str:
        return (
            f"You control a {ROBOT_TYPE} named '{ROBOT_NAME}' in a restaurant food court. "
            "Use the 'hover' tool to move the drone to specified 3D positions and the "
            "'describe_screen' tool to analyze the scene via the bottom camera. "
            "Report task status updates when completing each action."
        )

    # --- Camera callback ---

    def image_callback(self, msg: CompressedImage):
        """Store the latest bottom-camera frame as base64 for VLM queries."""
        try:
            self.latest_image_b64 = base64.b64encode(msg.data).decode("utf-8")
        except Exception as e:
            self.get_logger().error(f"Error converting image: {e}")

    # --- VLM / Scene description ---

    def query_callback(self, prompt: str) -> bool:
        """Send the latest camera frame + prompt to the VLM and publish the answer."""
        self.check_cancelled()

        if self.latest_image_b64 is None:
            self.get_logger().warn("No image received yet. Cannot query VLM.")
            return False

        self.get_logger().info("Sending image to VLM...")

        system_prompt = (
            "The table numbers are from left to right 1 to 4. "
            "The stall numbers are from left to right 1 to 3. "
            "You are a vision-based event detection assistant for a Drone. "
            "Your job is to analyze the image from the drone's bottom camera "
            "and answer the user's questions based on the visual content. "
            "IN SHORT, CONCISE ANSWERS ONLY. "
        )

        try:
            response = self.Visionclient.chat.completions.create(
                model="gpt-4.1",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{self.latest_image_b64}"
                                },
                            },
                        ],
                    },
                ],
            )

            answer = response.choices[0].message.content
            self.get_logger().info(f"VLM Answer: {answer}")
            answer = f"Drone (msg) | {answer}"
            self.pub_input_msg.publish(String(data=answer))
            return True

        except Exception as e:
            self.get_logger().error(f"OpenAI VLM request failed: {e}")
            return False

    # --- 3D movement ---

    def goto_service(self, x: float, y: float, z: float, yaw_deg: float) -> bool:
        """Call the drone position-controller service and wait for result."""
        self.check_cancelled()
        self.get_logger().info(f"Sending drone goto goal: x={x}, y={y}, z={z}, yaw={yaw_deg}°")

        if not self._goto_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("GotoPoseDrone service NOT available!")
            return False

        req = GotoPoseDrone.Request()
        req.x = x
        req.y = y
        req.z = z
        req.yaw_deg = yaw_deg

        future = self._goto_client.call_async(req)
        self.get_logger().info("Waiting for drone service response...")

        deadline = time.time() + 1000000.0
        while rclpy.ok() and not future.done():
            self.check_cancelled()
            time.sleep(0.05)
            if time.time() > deadline:
                self.get_logger().error("Drone goto service call TIMED OUT.")
                return False

        try:
            res = future.result()
        except Exception as e:
            self.get_logger().error(f"Drone goto service call failed: {e}")
            return False

        if not res.accepted:
            self.get_logger().warn(f"Drone goto request was NOT accepted: {res.message}")
            return False

        if res.success:
            self.get_logger().info(f"Drone goto SUCCESS: {res.message}")
            return True
        else:
            self.get_logger().warn(f"Drone goto FAILED: {res.message}")
            return False

    def hover(self, x: float = 0.0, y: float = 0.0, z: float = 2.0, yaw_deg: float = 0.0):
        """Move the drone to a 3D position with full state tracking."""
        self.get_logger().info(f"Hovering at position x={x}, y={y}, z={z}, yaw={yaw_deg}...")

        self.update_robot_state({
            "current_task": "hover",
            "task_status": "in_progress",
            "hover_stage": "started",
            "target_position": {"x": x, "y": y, "z": z, "yaw": yaw_deg},
        })

        self.update_robot_state({
            "task_status": "moving_to_position",
            "hover_stage": "navigating",
            "target_coords": {"x": x, "y": y, "z": z, "yaw_deg": yaw_deg},
        })

        success = self.goto_service(x=x, y=y, z=z, yaw_deg=yaw_deg)

        if success:
            self.get_logger().info(f"Hovering at position x={x}, y={y}, z={z} completed.")
            self.update_robot_state({
                "task_status": "completed",
                "hover_stage": "hovering",
                "current_position": {"x": x, "y": y, "z": z, "yaw": yaw_deg},
                "hover_successful": True,
                "hovering_at": f"({x}, {y}, {z})",
                "completion_timestamp": time.time(),
            })
            self.robot_task_completed(f"hover at x={x}, y={y}, z={z}")
        else:
            self.get_logger().error(f"Failed to reach hover position.")
            self.update_robot_state({
                "task_status": "failed",
                "hover_stage": "navigation_failed",
                "failure_reason": "could_not_reach_target_position",
                "hover_successful": False,
                "attempted_position": {"x": x, "y": y, "z": z, "yaw": yaw_deg},
            })
            self.robot_task_interrupted(f"hover at x={x}, y={y}, z={z}")

    def describe_screen(self, prompt: str = "What is in front of the drone?"):
        """Analyze the drone's bottom camera view using VLM with full state tracking."""
        self.get_logger().info(f'Describing screen with prompt: "{prompt}"...')

        self.update_robot_state({
            "current_task": "describe_screen",
            "task_status": "in_progress",
            "vision_query": prompt,
            "query_stage": "started",
        })

        if self.latest_image_b64 is None:
            self.get_logger().warn("No image received yet. Cannot query VLM.")
            self.update_robot_state({
                "task_status": "failed",
                "failure_reason": "no_image_available",
                "query_stage": "failed",
            })
            self.robot_task_interrupted(f"describe screen with prompt: {prompt}")
            return

        self.update_robot_state({
            "task_status": "querying_vlm",
            "query_stage": "processing_image",
            "image_available": True,
        })

        success = self.query_callback(prompt)

        if success:
            self.get_logger().info("Screen description completed.")
            self.update_robot_state({
                "task_status": "completed",
                "query_stage": "completed",
                "last_query": prompt,
                "query_successful": True,
                "completion_timestamp": time.time(),
            })
            self.robot_task_completed(f"describe screen with prompt: {prompt}")
        else:
            self.get_logger().error("Failed to get VLM response.")
            self.update_robot_state({
                "task_status": "failed",
                "query_stage": "vlm_error",
                "failure_reason": "vlm_query_failed",
                "query_successful": False,
            })
            self.robot_task_interrupted(f"describe screen with prompt: {prompt}")

    def stop_tasks(self):
        """Cancel all drone tasks and publish a cancel message to the position controller."""
        self.get_logger().info("Stopping all robot tasks...")
        self._task_cancelled = True
        msg = Bool()
        msg.data = True
        self._cancel_goto_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = DroneWorker()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
