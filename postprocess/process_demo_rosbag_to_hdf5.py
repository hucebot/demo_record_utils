"""
Script to extract desired data from recorded rosbags and store in HDF5 file.

Example usage: python process_demo_rosbag_to_hdf5.py --folder /path/to/folder
"""

import h5py
import pathlib
import time
from os import walk

from utils import (
    extract_compressed,
    extract_pose_stamped,
    extract_gripper_from_point_stamped,
    extract_joint_state,
    get_last_data_at_ref_times,
)


def main(dataset_name):
    start_time = time.time()

    filenames = next(walk(pathlib.Path(dataset_name).resolve()), (None, None, []))[2]
    num_bags = len(filenames)

    for demo_idx, demo_file in enumerate(filenames):
        print(f"Processing demo {demo_idx + 1}/{num_bags}")
        bagpath = pathlib.Path(dataset_name, demo_file).resolve()

        # Open rosbag and extract data.
        head_camera_color_times, head_camera_color_images = extract_compressed(
            bagpath, "/tiago_head_camera/color/image_raw/compressed"
        )
        right_camera_color_times, right_camera_color_images = extract_compressed(
            bagpath, "/tiago_right_camera/color/image_raw/compressed"
        )
        cmd_right_pose_times, cmd_right_pose_array = extract_pose_stamped(
            bagpath, "/dxl_input/pos_right"
        )
        cmd_right_gripper_times, cmd_right_gripper_array = (
            extract_gripper_from_point_stamped(bagpath, "/dxl_input/gripper_right")
        )

        goal_right_pose_times, goal_right_pose_array = extract_pose_stamped(
            bagpath, "/gripper_right_grasping_frame/goal"
        )
        read_right_pose_times, read_right_pose_array = extract_pose_stamped(
            bagpath, "/gripper_right_grasping_frame/read"
        )
        right_camera_right_pose_times, right_camera_right_pose_array = (
            extract_pose_stamped(
                bagpath, "/tiago_right_camera_color_optical_frame/pose"
            )
        )
        joint_times, joint_positions, joint_velocities = extract_joint_state(
            bagpath, "/joint_states"
        )

        # Synch data with head_camera_color timestamps.
        synch_head_camera_color_array = head_camera_color_images
        synch_right_camera_color_array = get_last_data_at_ref_times(
            head_camera_color_times, right_camera_color_times, right_camera_color_images
        )
        synch_cmd_right_pose_array = get_last_data_at_ref_times(
            head_camera_color_times, cmd_right_pose_times, cmd_right_pose_array
        )
        synch_cmd_right_gripper_array = get_last_data_at_ref_times(
            head_camera_color_times, cmd_right_gripper_times, cmd_right_gripper_array
        )
        synch_read_right_pose_array = get_last_data_at_ref_times(
            head_camera_color_times, read_right_pose_times, read_right_pose_array
        )
        synch_goal_right_pose_array = get_last_data_at_ref_times(
            head_camera_color_times, goal_right_pose_times, goal_right_pose_array
        )
        synch_right_camera_pose_array = get_last_data_at_ref_times(
            head_camera_color_times,
            right_camera_right_pose_times,
            right_camera_right_pose_array,
        )
        synch_joint_positions_array = get_last_data_at_ref_times(
            head_camera_color_times, joint_times, joint_positions
        )
        synch_joint_velocities_array = get_last_data_at_ref_times(
            head_camera_color_times, joint_times, joint_velocities
        )

        timestamps = head_camera_color_times - head_camera_color_times[0]
        timestamps = timestamps * 1e-3
        timestamps = timestamps.astype("float32")

        # Store demo data in h5 file with compression.
        demo_label = f"{demo_idx:03d}"
        with h5py.File(f"{dataset_name}.h5", "a") as h5file:
            demo_start_time = time.time()
            group = h5file.create_group(demo_label, track_order=True)
            group.create_dataset("timestamps", data=timestamps)
            group.create_dataset(
                "actions/cmd_right_pos", data=synch_cmd_right_pose_array[:, :3]
            )
            group.create_dataset(
                "actions/cmd_right_quat", data=synch_cmd_right_pose_array[:, 3:]
            )
            group.create_dataset(
                "actions/cmd_right_grip", data=synch_cmd_right_gripper_array
            )
            group.create_dataset(
                "observations/images/cam_head_color",
                data=synch_head_camera_color_array,
            )
            group.create_dataset(
                "observations/images/cam_right_wrist_color",
                data=synch_right_camera_color_array,
            )
            group.create_dataset(
                "observations/read_right_pos", data=synch_read_right_pose_array[:, :3]
            )
            group.create_dataset(
                "observations/read_right_quat", data=synch_read_right_pose_array[:, 3:]
            )
            group.create_dataset(
                "observations/goal_right_pos", data=synch_goal_right_pose_array[:, :3]
            )
            group.create_dataset(
                "observations/goal_right_quat", data=synch_goal_right_pose_array[:, 3:]
            )
            group.create_dataset(
                "observations/cam_right_wrist_pos",
                data=synch_right_camera_pose_array[:, :3],
            )
            group.create_dataset(
                "observations/cam_right_wrist_quat",
                data=synch_right_camera_pose_array[:, 3:],
            )
            group.create_dataset(
                "observations/joint_pos", data=synch_joint_positions_array
            )
            group.create_dataset(
                "observations/joint_vel", data=synch_joint_velocities_array
            )

            print(f"     data saved as demo '{demo_label}' in '{dataset_name}.h5' file")
            print(f"     time: {(time.time() - demo_start_time):.2f} seconds")

    print(f"Total time: {(time.time() - start_time):.2f} seconds")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert demo dataset from rosbag to hdf5"
    )
    parser.add_argument(
        "--folder", metavar="path", required=True, help="name of the dataset folder"
    )
    args = parser.parse_args()
    main(dataset_name=args.folder)
