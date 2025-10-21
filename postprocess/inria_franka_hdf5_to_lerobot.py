"""
Script to convert Inria hdf5 data to the LeRobot dataset v2.1 format.

Example usage: python inria_franka_hdf5_to_lerobot.py --hdf5_path /path/to/raw/data --repo_id <org>/<dataset-name> --task <task-name>
"""

import dataclasses
from pathlib import Path
import shutil
from typing import Literal
import imageio

import h5py
from lerobot.common.constants import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import matplotlib.pyplot as plt
import torch
import tqdm
import tyro
import yaml

import subprocess
import os
from contextlib import redirect_stdout, redirect_stderr


os.environ["IMAGEIO_FFMPEG_EXE"] = "/usr/local/bin/ffmpeg" # make packages compile with the libsvtav installed from source

def suppress_c_stdout_stderr():
    devnull = os.open(os.devnull, os.O_WRONLY)
    old_stdout = os.dup(1)
    old_stderr = os.dup(2)
    os.dup2(devnull, 1)
    os.dup2(devnull, 2)
    return old_stdout, old_stderr, devnull

def restore_c_stdout_stderr(old_stdout, old_stderr, devnull):
    os.dup2(old_stdout, 1)
    os.dup2(old_stderr, 2)
    os.close(old_stdout)
    os.close(old_stderr)
    os.close(devnull)


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 0.0001
    image_writer_processes: int = 10
    image_writer_threads: int = 5
    video_backend: str | None = None


DEFAULT_DATASET_CONFIG = DatasetConfig()


def create_empty_dataset(
    repo_id: str,
    robot_type: str,
    root: Path = HF_LEROBOT_HOME,
    mode: Literal["video", "image"] = "video",
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
    custom_config=None,
    cam_ft_dims=None,
) -> LeRobotDataset:
    fps_used = custom_config["fps_used"]

    action = custom_config["action"]["lerobot_names"]
    state = custom_config["state"]["lerobot_names"]

    cameras = [k for k in custom_config["cameras"].keys()]

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(state),),
            "names": {
                custom_config["state"]["lerobot_name"]: state,
            },
        },
        "action": {
            "dtype": "float32",
            "shape": (len(action),),
            "names": {
                custom_config["action"]["lerobot_name"]: action,
            },
        },
    }

    for cam in cameras:
        features[f"observation.images.{cam}"] = {
            "dtype": mode,
            "shape": cam_ft_dims[cam],
            "names": [
                "height",
                "width",
                "channels",
            ],
        }

    if Path(root / repo_id).exists():
        shutil.rmtree(root / repo_id)

    return LeRobotDataset.create(
        repo_id=repo_id,
        root=root / repo_id,
        fps=fps_used,
        robot_type=robot_type,
        features=features,
        use_videos=dataset_config.use_videos,
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
        video_backend=dataset_config.video_backend,
    )


def load_raw_images_per_camera(
    hd5_file: h5py.File, ep: int, camera_dict: dict[str, str]
) -> dict[str, np.ndarray]:
    imgs_per_cam = {}
    for camera in camera_dict.keys():
        uncompressed = hd5_file[f"{ep:03d}/"+camera_dict[camera]].ndim == 4

        if uncompressed:
            # load all images in RAM
            imgs_array = hd5_file[f"{ep:03d}/"+camera_dict[camera]][:]
        else:
            import cv2

            # load one compressed image after the other in RAM and uncompress
            imgs_array = []
            for data in hd5_file[f"{ep:03d}/"+camera_dict[camera]]:
                img = cv2.imdecode(data, cv2.IMREAD_COLOR)
                imgs_array.append(img[:, :, [2, 1, 0]])  # from BGR to RGB
            imgs_array = np.array(imgs_array)

        imgs_per_cam[camera] = imgs_array
    return imgs_per_cam


def load_raw_episode_data(
    hdf5_path: Path,
    ep: int,
    action_ft_list,
    state_ft_list,
    camera_dict,
    show_data_analysis,
    verbose,
) -> tuple[
    dict[str, np.ndarray],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    with h5py.File(hdf5_path, "r") as file:
        all_keys = []
        
        file.visit(all_keys.append)

        #print(all_keys)

        for key in all_keys:
            split_key = key.split("/")
            if len(split_key) < 3:
                continue
            
            #print(key, file[key][:].shape)

        #print(file["000/observations/f_ext"][:])

        # action
        action = []
        for action_ft_name in action_ft_list:
            action.append(torch.from_numpy(file[f"{ep:03d}/"+action_ft_name][:]))

        # state
        state = []
        for state_ft_name in state_ft_list:
            state.append(torch.from_numpy(file[f"{ep:03d}/"+state_ft_name][:]))

        if show_data_analysis:

            for i, state_ft_name in enumerate(state_ft_list):
                dists = np.linalg.norm(np.diff(state[i], 1, axis=0), axis=1)
                plt.plot(dists)
                plt.savefig(state_ft_name.split("/")[-1]+".png", bbox_inches="tight")
                plt.close()
            if verbose:
                print("Data analysis plot saved as plot.png")

        # camera
        imgs_per_cam = load_raw_images_per_camera(
            file,
            ep,
            camera_dict,
        )

        action = np.concatenate(action, 1)
        state = np.concatenate(state, 1)

    return (
        imgs_per_cam,
        state,
        action,
    )


def populate_dataset(
    dataset: LeRobotDataset,
    hdf5_path: Path,
    task: str,
    episodes: list[int] | None = None,
    custom_config=None,
    show_data_analysis=False,
    verbose=False,
) -> LeRobotDataset:
    
    for ep in tqdm.tqdm(episodes):

        (
            imgs_per_cam,
            state,
            action,
        ) = load_raw_episode_data(hdf5_path, ep, custom_config["action"]["hdf5_selected_names"], custom_config["state"]["hdf5_selected_names"], custom_config["cameras"], show_data_analysis, verbose)
        num_frames = state.shape[0]

        test_frames = {camera:[] for camera in imgs_per_cam.keys()}
        for i in range(num_frames):
            frame = {
                "observation.state": state[i],
                "action": action[i],
            }

            for camera, img_array in imgs_per_cam.items():
                test_frames[camera].append(img_array[i])
                frame[f"observation.images.{camera}"] = img_array[i]

            frame["task"] = task

            dataset.add_frame(frame)

        #for camera in test_frames.keys():
        #    imageio.mimwrite(camera+".mp4", test_frames[camera], codec="libsvtav1", ffmpeg_params=["-pix_fmt", "yuv420p","-movflags", "+faststart","-brand", "mp42","-strict", "experimental"], fps=30, macro_block_size=1)
        if verbose:
            dataset.save_episode()
        else:
            old_stdout, old_stderr, devnull = suppress_c_stdout_stderr()
            try:
                dataset.save_episode()
            finally:
                restore_c_stdout_stderr(old_stdout, old_stderr, devnull)

    return dataset


def port_inria_franka(
    hdf5_folder_path: Path,
    repo_id: str,
    hdf5_dataset_name: str | None = None,
    task: str = "DEBUG",
    *,
    episodes: list[int] | None = None,
    push_to_hub: bool = False,
    mode: Literal["video", "image"] = "video",
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
    show_data_analysis: bool = False,
    verbose=False,
):
    if (HF_LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(HF_LEROBOT_HOME / repo_id)

    config = None
    with open(hdf5_folder_path / "config.yaml") as f:
        config = yaml.safe_load(f)

    if hdf5_dataset_name is None:
        for f in os.listdir(hdf5_folder_path):
            if f.split(".")[-1] == "h5":
                hdf5_dataset_name = f
                break

    # Computes the feature dimensions and creates episodes if none selected
    f = h5py.File(hdf5_folder_path / hdf5_dataset_name, "r")
    if episodes is None:
        episodes = []
        for key in f.keys():
            episodes.append(int(key))
    imgs_per_cam = load_raw_images_per_camera(f, episodes[0], config["cameras"])
    cam_resolutions = {cam:imgs_per_cam[cam].shape[1:] for cam in imgs_per_cam.keys()}

    dataset = create_empty_dataset(
        repo_id,
        robot_type="franka",
        mode=mode,
        dataset_config=dataset_config,
        custom_config=config,
        cam_ft_dims=cam_resolutions,
    )
    dataset = populate_dataset(
        dataset,
        hdf5_folder_path / hdf5_dataset_name,
        task=task,
        episodes=episodes,
        custom_config=config,
        show_data_analysis=show_data_analysis,
        verbose=verbose,
    )

    subprocess.run(["chmod", "-R", "777", HF_LEROBOT_HOME], check=True)

    if push_to_hub:
        dataset.push_to_hub()


if __name__ == "__main__":
    tyro.cli(port_inria_franka)

    # example call:
    # $ python inria_tiago_hdf5_to_lerobot.py --hdf5_path place_fruit_in_bowl.h5 --repo_id /franka/place_fruit_in_bowl --task tiago_place_fruit_in_bowl

