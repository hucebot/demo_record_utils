"""
Script to extract franka (or other robot) manipulation demo from recorded rosbags and store in HDF5 file.

Example usage: python inria_franka_rosbag_to_hdf5.py --folder /path/to/folder
"""

import pathlib
import time
from os import walk
import yaml

from utils import (
    create_task,
    create_default_config,
)

import subprocess


def main(dataset_name, desired_dir, tasks_to_convert):
    start_time = time.time()

    # Read config file to get the selected topics
    dataset_path = pathlib.Path(dataset_name)
    config = None
    with open(dataset_path / "config.yaml", "r") as config_file:
        config = yaml.safe_load(config_file)

    topic_names = []
    hdf5_names = []
    topic_args = []
    for topic in config["selected_topics_conversion"]:
        topic_args.append({})
        for key in topic.keys():
            if key == "rosbag2_bag_topic_name":
                topic_names.append(topic[key])
            elif key == "hdf5_corresponding_name":
                hdf5_names.append(topic[key])
            else:
                topic_args[-1][key] = topic[key]

    reference_topic_name = config["reference_topic_name"]

    # Create new directory for hdf5
    desired_path = pathlib.Path(desired_dir)
    if desired_path.exists() == False:
        desired_path.mkdir(parents=True, exist_ok=True)

    # metadata necessary for the default config
    infos = []
    fps_tot_mean = {}

    # Create one hdf5 dataset for each task independently
    for task in tasks_to_convert:
        #print(topic_names)
        infos, fps_mean_task = create_task(dataset_path, desired_path, task, reference_topic_name, topic_names, hdf5_names, topic_args)
        fps_tot_mean[task] = fps_mean_task

    # Create default config if there is none
    if (desired_path / "config.yaml").exists() == False:
        create_default_config(fps_tot_mean, infos, desired_path / "config.yaml")

    subprocess.run(["chmod", "-R", "777", desired_path], check=True)

    print(f"Total time: {(time.time() - start_time):.2f} seconds")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert demo dataset from rosbag to hdf5"
    )
    parser.add_argument(
        "--rosbag_folder", required=True, help="name of the rosbag dataset folder"
    )
    parser.add_argument("--hdf5_dir", default="", help="name of the desired hdf5 dataset directory")
    parser.add_argument("--tasks", default=[], help="names of the tasks to convert", nargs='+')
    args = parser.parse_args()

    desired_dir = args.hdf5_dir
    if desired_dir == "": # the rosbag dataset folder name is the desired directory for hdf5 datasets by default
        desired_dir = args.rosbag_folder

    tasks = args.tasks
    if len(tasks) == 0:
        tasks = next(walk('./'+args.rosbag_folder))[1]

    main(dataset_name=args.rosbag_folder, desired_dir=desired_dir, tasks_to_convert=tasks)
