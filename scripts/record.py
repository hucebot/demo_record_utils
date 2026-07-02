#!/usr/bin/env python3
import os
import sys
import argparse
import rclpy

# Import your robot-specific classes
from streamdeck.streamdeck_franka import StreamDeckFranka
from streamdeck.streamdeck_tiago import StreamDeckTiago

# === CENTRALIZED ROBOT CONFIGURATIONS ===
#TODO: maybe use dataclass configs so that we can easily modify those with tyro?
ROBOT_CONFIGS = {
    "franka": {
        "demo_name": "demo_task1",
        "bag_base_dir": "/datasets",
        "topics": [
            "/camera/camera/color/camera_info",
            "/camera/camera/color/image_raw/compressed",
            "/camera/camera/depth/camera_info",
            "/camera/camera/depth/image_rect_raw",
            "/webcam1/image_raw/compressed",
            "/webcam2/image_raw/compressed",
            "/webcam3/image_raw/compressed",
            "/cartesian_impedance/cartesian_pos_curr",
            "/cartesian_impedance/cartesian_pos_des_filt",
            "/cartesian_impedance/equilibrium_pose",
            "/bota_ft_sensor/wrench_filtered",
            "/cartesian_impedance/joint_state",
            "/panda_gripper/gripper_command",
            "/panda_gripper/width"
        ]
    },
    "tiago": {
        "demo_name": "tiago_task1",
        "bag_base_dir": "/datasets",
        "topics": [#TODO: fill in the topics for tiago
        ]
    }
}

def main():
    parser = argparse.ArgumentParser(description="Launch Stream Deck Controller.")
    parser.add_argument(
        '--robot',
        type=str,
        required=True,
        choices=ROBOT_CONFIGS.keys(),
        help="Specify which robot to load"
    )
    parser.add_argument(
        '--task',
        type=str,
        required=False,
        help="Override the default demo_name for this run"
    )

    # Separate our args from standard ROS args
    args, ros_args = parser.parse_known_args(sys.argv[1:])

    # Load config and apply task override if provided
    config = ROBOT_CONFIGS[args.robot]
    if args.task:
        config["demo_name"] = args.task

    rclpy.init(args=ros_args)

    # Instantiate the correct class based on the robot argument
    if args.robot == "franka":
        node = StreamDeckFranka(config)
    elif args.robot == "tiago":
        node = StreamDeckTiago(config)

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