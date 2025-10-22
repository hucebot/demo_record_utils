"""
Script to extract franka (or other robot) manipulation demo from recorded rosbags and store in HDF5 file.

Example usage: python inria_franka_rosbag_to_hdf5.py --folder /path/to/folder
"""

import h5py
import pathlib
import time
from os import walk
import yaml
import numpy as np

from utils import (
    extractCompressedImage,
    extractPoseStamped,
    extractGripperWidthFromGripperWidth,
    extractJointState,
    getLastDataAtRefTimes,
)

import subprocess

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def extract_topic(topic_type, bagpath, topic_name, topic_args):
    if topic_type == "sensor_msgs/msg/CompressedImage":
        return extractCompressedImage(bagpath, topic_name, **topic_args)
    elif topic_type == "sensor_msgs/msg/JointState":
        return extractJointState(bagpath, topic_name, **topic_args)
    elif topic_type == "geometry_msgs/msg/PoseStamped":
        return extractPoseStamped(bagpath, topic_name, **topic_args)
    elif topic_type == "custom_msgs/msg/GripperWidth":
        return extractGripperWidthFromGripperWidth(bagpath, topic_name, **topic_args)
    else:
        raise NotImplementedError
    
def create_default_config(fps_used, infos, dir):
    
    # Default selection of topics in three categories : actions, states, cameras
    action_name, state_name = "motor", "motor"
    action_names, state_names, cameras_names = [], [], []
    action_dims, state_dims, cameras_new_names = [], [], []
    for v in infos:
        hdf5_name, ft_dim = v["hdf5_name"], v["ft_dim"]

        if "action" in hdf5_name:
            action_names.append(hdf5_name)
            action_dims.append(ft_dim)
            if "joint" in hdf5_name:
                action_name = "joint"
        elif "cam" in hdf5_name:
            cameras_names.append(hdf5_name)
            cameras_new_names.append(hdf5_name.split("/")[-1])
        else:
            state_names.append(hdf5_name)
            state_dims.append(ft_dim)
            if "joint" in hdf5_name:
                state_name = "joint"

    with open(dir, "w") as config_file:
        action_names_fts = []
        for i, act_name in enumerate(action_names):
            action_names_fts += [act_name.split("/")[-1]+"_"+str(j) for j in range(action_dims[i])]

        state_names_fts = []
        for i, state_name in enumerate(state_names):
            state_names_fts += [state_name.split("/")[-1]+"_"+str(j) for j in range(state_dims[i])]

        new_config_dico = {
            "fps_used":fps_used,
            "action":{
                "lerobot_name": action_name, 
                "lerobot_names": action_names_fts, 
                "hdf5_selected_names":action_names 
                }, 
            "state":{
                "lerobot_name": state_name, 
                "lerobot_names": state_names_fts, 
                "hdf5_selected_names":state_names
                },  
            "cameras":{new_cam_name: cam_name for new_cam_name, cam_name in zip(cameras_new_names, cameras_names)}
        }
        yaml.dump(new_config_dico, config_file, default_flow_style=False)

def main(dataset_name, desired_dir):
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

    number_topics = len(topic_names)

    # Read metadata file to get each topic and its type
    topic_types = {}
    with open(dataset_path / "metadata.yaml", "r") as metadata_file:
        metadata = yaml.safe_load(metadata_file)
        
        for topic_info in metadata["rosbag2_bagfile_information"]["topics_with_message_count"]:
            topic_types[topic_info["topic_metadata"]["name"]] = topic_info["topic_metadata"]["type"]

    filenames = sorted(next(walk((dataset_path / "data").resolve()), (None, None, []))[2])
    num_bags = len(filenames)

    # Create new directory for hdf5
    desired_path = pathlib.Path(desired_dir)
    if desired_path.exists() == False:
        desired_path.mkdir(parents=True, exist_ok=True)

    infos = []

    fps_tot_mean = 0
    for demo_idx, demo_file in enumerate(filenames):
        print(f"Processing demo {demo_idx + 1}/{num_bags}")
        bagpath = pathlib.Path(dataset_name, "data", demo_file).resolve()
        demo_label = f"{demo_idx:03d}"
        demo_start_time = time.time()

        # get the infos if it is the first demonstration
        if demo_idx == 0:
            for i, topic_name in enumerate(topic_names):
                topic_times, topic_data = extract_topic(topic_types[topic_name], bagpath, topic_name, topic_args[i])
                infos.append({})
                infos[-1]["hdf5_name"] = hdf5_names[i]
                infos[-1]["ft_dim"] = topic_data.shape[1]

        # search for reference topic
        reference_topic = {}
        for i, topic_name in enumerate(topic_names):
            if topic_name == reference_topic_name:
                topic_times, topic_data = extract_topic(topic_types[topic_name], bagpath, topic_name, topic_args[i])
                reference_topic["times"] = topic_times
                reference_topic["data"] = topic_data
                break
            elif i == number_topics-1:
                if demo_idx == 0:
                    print(f"{bcolors.WARNING}Warning : The given reference topic is unavailable in the data so the last topic is considered the reference topic by default !{bcolors.ENDC}")

                topic_times, topic_data = extract_topic(topic_types[topic_name], bagpath, topic_name, topic_args[i])
                reference_topic["times"] = topic_times
                reference_topic["data"] = topic_data

        timestamps = reference_topic["times"] - reference_topic["times"][0]
        timestamps = timestamps * 1e-3
        timestamps = timestamps.astype("float32")

        #print(timestamps)
        fps_mean, fps_std = np.mean(1/np.diff(timestamps[:,0])).item(), np.std(1/np.diff(timestamps[:,0])).item()
        print("fps", fps_mean, "+/-", fps_std)
        fps_tot_mean = (fps_tot_mean*demo_idx+fps_mean)/(demo_idx+1)



        with h5py.File(f"{desired_path / dataset_name}.h5", "a") as h5file:
            group = h5file.create_group(demo_label, track_order=True)
            group.create_dataset("timestamps", data=timestamps)

            for topic_idx, topic_name in enumerate(topic_names):
                # Open rosbag and extract topic data.
                result = extract_topic(topic_types[topic_name], bagpath, topic_name, topic_args[topic_idx])
                #print(topic_name, len(result))
                topic_times, topic_data = result

                # Synch data with given topic timestamps.
                synch_topic_data = getLastDataAtRefTimes(reference_topic["times"], topic_times, topic_data)

                group.create_dataset(hdf5_names[topic_idx], data=synch_topic_data)

        print(f"     data saved as demo '{demo_label}' in '{desired_path / dataset_name}.h5' file")
        print(f"     time: {(time.time() - demo_start_time):.2f} seconds")

    print(f"Total time: {(time.time() - start_time):.2f} seconds")

    create_default_config(fps_tot_mean, infos, desired_path / "config.yaml")

    subprocess.run(["chmod", "-R", "777", desired_path], check=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert demo dataset from rosbag to hdf5"
    )
    parser.add_argument(
        "--rosbag_folder", required=True, help="name of the rosbag dataset folder"
    )
    parser.add_argument("--hdf5_dir", default="", help="name of the desired hdf5 dataset directory")
    args = parser.parse_args()

    desired_dir = args.hdf5_dir
    if desired_dir == "":
        desired_dir = args.rosbag_folder
    main(dataset_name=args.rosbag_folder, desired_dir=desired_dir)
