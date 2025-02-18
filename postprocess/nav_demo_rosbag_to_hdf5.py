"""
Script to extract manipulation demo from recorded rosbags and store in HDF5 file.

Example usage: python manip_demo_rosbag_to_hdf5.py --folder /path/to/folder
"""

import h5py
import pathlib
import time
from os import walk

from utils import (
    extractAndEncodeImage,
    extractTwist,
    getLastDataAtRefTimes,
)


def main(dataset_name):
    start_time = time.time()

    filenames = next(walk(pathlib.Path(dataset_name).resolve()), (None, None, []))[2]
    num_bags = len(filenames)

    for demo_idx, demo_file in enumerate(filenames):
        print(f"Processing demo {demo_idx + 1}/{num_bags}")
        bagpath = pathlib.Path(dataset_name, demo_file).resolve()

        # Open rosbag and extract data.
        head_camera_color_times, head_camera_color_images = extractAndEncodeImage(
            bagpath,
            "/tiago_head_camera/color/image_raw",
        )
        head_camera_depth_times, head_camera_depth_images = extractAndEncodeImage(
            bagpath,
            "/tiago_head_camera/depth/image_raw",
        )
        cmd_vel_times, cmd_vel_array = extractTwist(
            bagpath,
            "/mobile_base_controller/cmd_vel",
        )

        # Synch data with head_camera_color timestamps.
        synch_head_camera_color_array = head_camera_color_images
        synch_head_camera_depth_array = getLastDataAtRefTimes(
            head_camera_color_times, head_camera_depth_times, head_camera_depth_images
        )
        synch_cmd_vel_array = getLastDataAtRefTimes(
            head_camera_color_times, cmd_vel_times, cmd_vel_array
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
                "actions/cmd_vel",
                data=synch_cmd_vel_array,
            )
            group.create_dataset(
                "observations/images/cam_head_color",
                data=synch_head_camera_color_array,
            )
            group.create_dataset(
                "observations/images/cam_head_depth",
                data=synch_head_camera_depth_array,
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
