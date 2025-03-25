"""
Script to convert Inria hdf5 data to the LeRobot dataset v2.0 format.

Example usage: python inria_tiago_hdf5_to_lerobot.py --hdf5_path /path/to/raw/data --repo_id <org>/<dataset-name> --task <task-name>
"""

import dataclasses
from pathlib import Path
import shutil
from typing import Literal

import h5py
from lerobot.common.constants import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import torch
import tqdm
import tyro


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
) -> LeRobotDataset:
    action = [
        "x",
        "y",
        "z",
        "qx",
        "qy",
        "qz",
        "qw",
        "grip",
    ]
    state = [
        "x",
        "y",
        "z",
        "qx",
        "qy",
        "qz",
        "qw",
    ]
    cameras = [
        "cam_head_color",
    ]

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(state),),
            "names": {
                "ee": state,
            },
        },
        "action": {
            "dtype": "float32",
            "shape": (len(action),),
            "names": {
                "ee": action,
            },
        },
    }

    for cam in cameras:
        features[f"observation.images.{cam}"] = {
            "dtype": mode,
            "shape": (720, 1280, 3),
            "names": [
                "height",
                "width",
                "channels",
            ],
        }

    if Path(HF_LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(HF_LEROBOT_HOME / repo_id)

    return LeRobotDataset.create(
        repo_id=repo_id,
        root=root,
        fps=30,
        robot_type=robot_type,
        features=features,
        use_videos=dataset_config.use_videos,
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
        video_backend=dataset_config.video_backend,
    )


def load_raw_images_per_camera(
    hd5_file: h5py.File, ep: int, cameras: list[str]
) -> dict[str, np.ndarray]:
    imgs_per_cam = {}
    for camera in cameras:
        uncompressed = hd5_file[f"{ep:03d}/observations/images/{camera}"].ndim == 4

        if uncompressed:
            # load all images in RAM
            imgs_array = hd5_file[f"{ep:03d}/observations/images/{camera}"][:]
        else:
            import cv2

            # load one compressed image after the other in RAM and uncompress
            imgs_array = []
            for data in hd5_file[f"{ep:03d}/observations/images/{camera}"]:
                img = cv2.imdecode(data, cv2.IMREAD_COLOR)
                imgs_array.append(img[:, :, [2, 1, 0]])  # from BGR to RGB
            imgs_array = np.array(imgs_array)

        imgs_per_cam[camera] = imgs_array
    return imgs_per_cam


def load_raw_episode_data(
    hdf5_path: Path,
    ep: int,
) -> tuple[
    dict[str, np.ndarray],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    with h5py.File(hdf5_path, "r") as file:
        # action
        cmd_right_pos = torch.from_numpy(file[f"{ep:03d}/actions/cmd_right_pos"][:])
        cmd_right_quat = torch.from_numpy(file[f"{ep:03d}/actions/cmd_right_quat"][:])
        cmd_right_grip = torch.from_numpy(file[f"{ep:03d}/actions/cmd_right_grip"][:])
        action = np.concatenate((cmd_right_pos, cmd_right_quat, cmd_right_grip), 1)

        # state
        read_right_pos = torch.from_numpy(
            file[f"{ep:03d}/observations/read_right_pos"][:]
        )
        read_right_quat = torch.from_numpy(
            file[f"{ep:03d}/observations/read_right_quat"][:]
        )
        state = np.concatenate((read_right_pos, read_right_quat), 1)

        imgs_per_cam = load_raw_images_per_camera(
            file,
            ep,
            [
                "cam_head_color",
            ],
        )

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
) -> LeRobotDataset:
    if episodes is None:
        f = h5py.File(hdf5_path, "r")
        episodes = []
        for key in f.keys():
            episodes.append(int(key))

    for ep in tqdm.tqdm(episodes):

        (
            imgs_per_cam,
            state,
            action,
        ) = load_raw_episode_data(hdf5_path, ep)
        num_frames = state.shape[0]

        for i in range(num_frames):
            frame = {
                "observation.state": state[i],
                "action": action[i],
            }

            for camera, img_array in imgs_per_cam.items():
                frame[f"observation.images.{camera}"] = img_array[i]

            frame["task"] = task

            dataset.add_frame(frame)

        dataset.save_episode()

    return dataset


def port_inria_tiago(
    hdf5_path: Path,
    repo_id: str,
    task: str = "DEBUG",
    *,
    episodes: list[int] | None = None,
    push_to_hub: bool = False,
    mode: Literal["video", "image"] = "video",
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
):
    if (HF_LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(HF_LEROBOT_HOME / repo_id)

    dataset = create_empty_dataset(
        repo_id,
        robot_type="tiago",
        mode=mode,
        dataset_config=dataset_config,
    )
    dataset = populate_dataset(
        dataset,
        hdf5_path,
        task=task,
        episodes=episodes,
    )
    dataset.consolidate()

    if push_to_hub:
        dataset.push_to_hub()


if __name__ == "__main__":
    tyro.cli(port_inria_tiago)

    # example call:
    # $ python inria_tiago_hdf5_to_lerobot.py --hdf5_path place_fruit_in_bowl.h5 --repo_id /tiago/place_fruit_in_bowl --task tiago_place_fruit_in_bowl

