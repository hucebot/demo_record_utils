#!/usr/bin/env python3

import rclpy
from std_srvs.srv import Trigger
from streamdeck.streamdeck_base import StreamDeckBase

class StreamDeckTiago(StreamDeckBase):
    def __init__(self, config):
        # Pass the config up to the base class so it can handle the recording
        super().__init__('stream_deck_tiago', 'tiago_buttons.json', config)

        # Tiago-specific ROS 2 interfaces (just Homing for this repo)
        # TODO: perhaps we need resume too, this needs to be checked once I finalize the ros2 tiago wbc repo
        self.cli_home = self.create_client(Trigger, '/home_position')

        self.get_logger().info(f"Tiago Stream Deck Initialized for: {self.demo_name}")

    # --- Tiago Specific Homing Logic ---
    def press_home(self, key_index):
        self.get_logger().info("Requesting Home Sequence from Cartesian Interface...")

        # Offload to ROS thread to avoid hardware freeze
        self._schedule_one_shot(0.01, lambda: self._do_home_request(key_index))

    def _do_home_request(self, key_index):
        if not self.cli_home.wait_for_service(timeout_sec=1.0):
            self.get_logger().error("Home service not available!")
            self.update_button_visual(key_index, "HOME\nERR", self.colors["error"], self.colors["text"])
            self._schedule_one_shot(1.0, lambda: self.press_default(key_index))
            return

        self.update_button_visual(key_index, "HOMING...", self.colors["active"], self.colors["text"])

        req = Trigger.Request()
        future = self.cli_home.call_async(req)
        future.add_done_callback(lambda f: self._on_home_response(f, key_index))

    def _on_home_response(self, future, key_index):
        try:
            response = future.result()
            if response.success:
                self.get_logger().info(f"Homing response: {response.message}")
            else:
                self.get_logger().warn(f"Homing failed: {response.message}")
        except Exception as e:
            self.get_logger().error(f"Home service call failed: {e}")

        # Reset visually after roughly the duration the hardware takes (e.g., 4.5s)
        self._schedule_one_shot(4.5, lambda: self.press_default(key_index))
