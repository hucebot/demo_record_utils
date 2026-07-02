"""
Script to extract franka (or other robot) manipulation demo from recorded rosbags and store in HDF5 file.

Example usage:

python inria_franka_rosbag_to_hdf5.py \
    --rosbag_folder /mnt/Data/rosbags \
    --hdf5_dir /mnt/Data/converted_dionisis \
    --tasks cubes

"""

import pathlib
import time
from os import walk
import yaml

from utils import (
    create_task,
    add_config,
)

import subprocess


# ADD config_path TO THE ARGUMENTS
def main(dataset_name, desired_dir, tasks_to_convert, config_path, verbose):
    start_time = time.time()

    dataset_path = pathlib.Path(dataset_name)


    with open(config_path, "r") as config_file:
        config = yaml.safe_load(config_file)

    selected_topics = {}
    for topic in config["selected_topics_conversion"]:
        topic_args = {}
        for key in topic.keys():
            if key != "from_rosbag_topic_name" and key != "hdf5_name":
                topic_args[key] = topic[key]

        if topic["from_rosbag_topic_name"] in selected_topics.keys():
            selected_topics[topic["from_rosbag_topic_name"]][topic["hdf5_name"]] = topic_args
        else:
            selected_topics[topic["from_rosbag_topic_name"]] = {topic["hdf5_name"]:topic_args}

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
        infos, fps_mean_task = create_task(dataset_path, desired_path, task, reference_topic_name, selected_topics, verbose=verbose)
        fps_tot_mean[task] = fps_mean_task

    # Create default config if there is none
    if (desired_path / "config.yaml").exists() == False:
        add_config(fps_tot_mean, infos, desired_path / "config.yaml", default=True)
    else:
        add_config(fps_tot_mean, infos, desired_path / "config.yaml")

    subprocess.run(["chmod", "-R", "777", desired_path], check=True)

    print(f"Total time: {(time.time() - start_time):.2f} seconds")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert demo dataset from rosbag to hdf5")
    parser.add_argument("--rosbag_folder", required=True, help="name of the rosbag dataset folder")
    parser.add_argument("--hdf5_dir", default="", help="name of the desired hdf5 dataset directory")
    parser.add_argument("--tasks", default=[], help="names of the tasks to convert", nargs='+')
    parser.add_argument("--verbose", action="store_true", help="name of the desired hdf5 dataset directory")
    parser.add_argument("--config", default="./postprocess/config_rosbag2hdf5/config.yaml", help="path to config.yaml")
    args = parser.parse_args()

    desired_dir = args.hdf5_dir
    if desired_dir == "": # the rosbag dataset folder name is the desired directory for hdf5 datasets by default
        desired_dir = args.rosbag_folder

    tasks = args.tasks
    if len(tasks) == 0:
        tasks = next(walk(args.rosbag_folder))[1]

    main(dataset_name=args.rosbag_folder, desired_dir=desired_dir, tasks_to_convert=tasks, verbose=args.verbose, config_path=args.config)
