#!/usr/bin/env python3

import rclpy
from std_srvs.srv import Trigger
from geometry_msgs.msg import PointStamped
from controller_manager_msgs.srv import SwitchController, LoadController, ConfigureController
from streamdeck.streamdeck_base import StreamDeckBase

class StreamDeckFranka(StreamDeckBase):
    def __init__(self, config):
        # Pass the config up to the base class so it can handle the recording
        super().__init__('stream_deck_franka', 'franka_buttons.json', config)


        # Franka specific clients needed for homing and taring the force-torque sensor
        self.cli_tare = self.create_client(Trigger, '/bota_ft_sensor/tare')
        self.cli_switch_controller = self.create_client(SwitchController, '/controller_manager/switch_controller')
        self.cli_load_controller = self.create_client(LoadController, '/controller_manager/load_controller')
        self.cli_configure_controller = self.create_client(ConfigureController, '/controller_manager/configure_controller')

        self.get_logger().info(f"Franka Stream Deck Initialized for: {self.demo_name}")

        # Re-render to ensure Franka-specific colors/labels are pushed to the deck
        self.render_all_buttons()

    # --- Franka Specific Homing Logic ---
    def press_home(self, key_index):
        self.get_logger().info("Starting Home Sequence...")
        self.update_button_visual(key_index, "HOMING...", self.colors["active"], self.colors["text"])
        self._schedule_one_shot(0.01, lambda: self._do_load_controller(key_index))

    def _do_load_controller(self, key_index):
        if not self.cli_load_controller.wait_for_service(timeout_sec=1.0):
            self.get_logger().error("Controller manager load service not available")
            self.press_default(key_index)
            return

        req = LoadController.Request()
        req.name = 'move_to_start_example_controller'
        future = self.cli_load_controller.call_async(req)
        future.add_done_callback(lambda f: self._on_home_controller_loaded(f, key_index))

    def _on_home_controller_loaded(self, future, key_index):
        try:
            future.result()
        except Exception as e:
            self.get_logger().error(f"Load service call failed: {e}")

        if not self.cli_configure_controller.wait_for_service(timeout_sec=1.0):
            self.get_logger().error("Controller manager configure service not available")
            self.press_default(key_index)
            return

        req = ConfigureController.Request()
        req.name = 'move_to_start_example_controller'
        future2 = self.cli_configure_controller.call_async(req)
        future2.add_done_callback(lambda f: self._on_home_controller_configured(f, key_index))

    def _on_home_controller_configured(self, future, key_index):
        try:
            future.result()
        except Exception as e:
            self.get_logger().error(f"Configure service call failed: {e}")

        self.get_logger().info("Switching to move_to_start_example_controller...")
        self._switch_controllers(['move_to_start_example_controller'], ['custom_cartesian_impedance_controller'])
        self._schedule_one_shot(5.0, lambda: self._finish_home(key_index))

    def _finish_home(self, key_index):
        self.get_logger().info("Home Sequence Complete. Reactivating Impedance Controller...")
        self._switch_controllers(['custom_cartesian_impedance_controller'], ['move_to_start_example_controller'])
        self.press_default(key_index)

    # --- Controller Switch Helper ---
    def _switch_controllers(self, activate_list, deactivate_list):
        if not self.cli_switch_controller.wait_for_service(timeout_sec=1.0):
            self.get_logger().error("Controller manager switch service not available")
            return False
        req = SwitchController.Request()
        req.activate_controllers = activate_list
        req.deactivate_controllers = deactivate_list
        req.strictness = SwitchController.Request.STRICT
        self.cli_switch_controller.call_async(req)
        return True

    def press_tare_fts(self, key_index):
        if not self.cli_tare.wait_for_service(timeout_sec=1.0):
            self.get_logger().error("Tare service not available")
            self.update_button_visual(key_index, "TARE\nERR", self.colors.get("error", "#FFB3BA"), self.colors["text"])
            self._schedule_one_shot(1.0, lambda: self.press_default(key_index))
            return

        self.cli_tare.call_async(Trigger.Request())
        self.press_default(key_index)


# Local testing block
def main(args=None):
    test_config = {
        "demo_name": "franka_test",
        "bag_base_dir": "./",
        "topics": ["/joint_states"]
    }
    rclpy.init(args=args)
    node = StreamDeckFranka(test_config)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()