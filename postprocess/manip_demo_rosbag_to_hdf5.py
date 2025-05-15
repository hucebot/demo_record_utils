# manip_demo_rosbag_to_hdf5.py

"""
Script to extract manipulation demo from recorded rosbags and store in HDF5 file.

Example usage: python manip_demo_rosbag_to_hdf5.py --folder /path/to/folder
"""

import h5py
import pathlib
import time
from os import walk
from tqdm import tqdm

from utils import (
    extractCompressedImage,
    extractPoseStamped,
    extractGripperFromPointStamped,
    extractJointState,
    extractTF,
    getLastDataAtRefTimes,
)


def rosbag_to_hdf5(dataset_name):
    start_time = time.time()

    filenames = next(walk(pathlib.Path(dataset_name).resolve()), (None, None, []))[2]
    num_bags = len(filenames)

    for demo_idx, demo_file in enumerate(tqdm(filenames, desc="Processing demo", unit="demo"), start=0):
        bagpath = pathlib.Path(dataset_name, demo_file).resolve()

        frontal_camera_color_times, frontal_camera_color_images = extractCompressedImage(
            bagpath, "/camera/color/compressed"
        )

        cmd_right_pose_times, cmd_right_pose_array = extractPoseStamped(
            bagpath, "/dxl_input/pos_right"
        )
        cmd_right_gripper_times, cmd_right_gripper_array = extractGripperFromPointStamped(
            bagpath, "/dxl_input/gripper_right"
        )

        cmd_tf_times, cmd_tf_array = extractTF(
            bagpath,
            "/tf",
            "ci/world",
            "ci/gripper_right_grasping_frame",
            verbose=True
        )

        read_right_pose_times, read_right_pose_array = extractPoseStamped(
            bagpath, "/cartesian/gripper_right_grasping_frame/reference"
        )

        # joint_times, joint_positions, joint_velocities = extractJointState(
        #     bagpath, "/joint_states"
        # )

        synch_head_camera_color_array = frontal_camera_color_images
        synch_cmd_right_pose_array = getLastDataAtRefTimes(
            frontal_camera_color_times, cmd_right_pose_times, cmd_right_pose_array
        )
        synch_cmd_right_gripper_array = getLastDataAtRefTimes(
            frontal_camera_color_times, cmd_right_gripper_times, cmd_right_gripper_array
        )
        synch_read_right_pose_array = getLastDataAtRefTimes(
            frontal_camera_color_times, read_right_pose_times, read_right_pose_array
        )

        synch_cmd_tf_array = getLastDataAtRefTimes(
            frontal_camera_color_times, cmd_tf_times, cmd_tf_array
        )

        # synch_joint_positions_array = getLastDataAtRefTimes(
        #     frontal_camera_color_times, joint_times, joint_positions
        # )

        # synch_joint_velocities_array = getLastDataAtRefTimes(
        #     frontal_camera_color_times, joint_times, joint_velocities
        # )


        timestamps = frontal_camera_color_times[:,0] - frontal_camera_color_times[0,0]
        timestamps = timestamps * 1e-3
        timestamps = timestamps.astype("float32")

        demo_label = f"{demo_idx:03d}"
        with h5py.File(f"{dataset_name}.h5", "a") as h5file:
            group = h5file.create_group(demo_label, track_order=True)
            group.create_dataset("timestamps", data=timestamps)


            group.create_dataset("actions/cmd_right_pos", data=synch_cmd_right_pose_array[:, :3])
            group.create_dataset("actions/cmd_right_quat", data=synch_cmd_right_pose_array[:, 3:])
            group.create_dataset("actions/cmd_right_grip", data=synch_cmd_right_gripper_array)

            group.create_dataset("observations/images/cam_head_color", data=synch_head_camera_color_array)
            group.create_dataset("observations/read_right_pos", data=synch_read_right_pose_array[:, :3])
            group.create_dataset("observations/read_right_quat", data=synch_read_right_pose_array[:, 3:])

            group.create_dataset("observations/cmd_tf_trans", data=synch_cmd_tf_array[:, :3])
            group.create_dataset("observations/cmd_tf_rot", data=synch_cmd_tf_array[:, 3:])

            # group.create_dataset("observations/joint_pos", data=synch_joint_positions_array)
            # group.create_dataset("observations/joint_vel", data=synch_joint_velocities_array)

    print(f"Total time: {(time.time() - start_time):.2f} seconds")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert demo dataset from rosbag to hdf5")
    parser.add_argument("--folder", required=True, help="name of the dataset folder")
    args = parser.parse_args()
    rosbag_to_hdf5(dataset_name=args.folder)
