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
import cv2
import sys


os.environ["IMAGEIO_FFMPEG_EXE"] = "/usr/local/bin/ffmpeg" # make packages compile with the libsvtav installed from source

# Silence utility functions (to avoid printing in the shell the excessive svt library warning stdouts)

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

# Data analysis utility functions

def spearman_coefficient_matrix(x, y):
    """
    Parameters :
    x: Array of shape (N, K)
    y: Array of shape (N, L)

    Returns :
    Spearman matrix: Array of shape (K, L)
    """
    rx, ry = np.argsort(x, axis=0), np.argsort(y, axis=0)

    mean_rx, mean_ry = np.mean(rx[:,:,None], axis=0), np.mean(ry[:,None,:], axis=0)
    cov_rxry = np.mean(rx[:,:,None]*ry[:,None,:], axis=0) - mean_rx*mean_ry
    var_rx, var_ry = np.mean(rx[:,:,None]**2, axis=0) - mean_rx**2, np.mean(ry[:,None,:]**2, axis=0) - mean_ry**2

    return cov_rxry/np.sqrt(var_ry*var_rx)

def find_lower_outlier(x):
    """
    Parameters :
    x : Array of positive floats and shape (N,)

    Returns :
    Index : the sorted indices of lower outliers in x
    """
    sorted_idx = np.argsort(x)
    sorted_x = x[sorted_idx]

    # dichotomy search for the lower outlier with the largest value 
    last_dec_idx, gamma = [0, 0], [0., 0.01]
    for j in range(10):
        gamma_pow = gamma[-1]**(np.arange(len(sorted_x))+1)
        soft_sorted_x = (1-gamma[-1])/(1-gamma_pow)*np.cumsum(gamma_pow*sorted_x)
        Femp = np.mean((soft_sorted_x[None,:] <= soft_sorted_x[:,None])*1., axis=1)
        Femp = np.clip((Femp[1:] - Femp[:-1])/(soft_sorted_x[1:]-soft_sorted_x[:-1]), 0., 100)

        # Select the last outlier index that affects heavily the distribution : the total count of outliers must not exceed 20% of the total data sample size
        new_last_dec_idx = 0
        for i in range(len(Femp)//5):
            if Femp[i+1] < Femp[i]:
                new_last_dec_idx = i+2
        if new_last_dec_idx - 1 < len(Femp)//5:
            last_dec_idx[0] = last_dec_idx[1]
            last_dec_idx[1] = new_last_dec_idx
            gamma[0] = gamma[1]
            gamma[1] = (gamma[1]+1)/2
        else:
            gamma[1] = (gamma[1]+gamma[0])/2

        print(new_last_dec_idx, len(Femp)//2, "gamma : ", gamma, "last idx : ", last_dec_idx)

    return np.sort(sorted_idx[:last_dec_idx[1]])

def process_image(data_enum, width, height, uncompressed):
    imgs_array = []
    if uncompressed:
        # load all images in RAM
        for data in data_enum:
            img = data/np.max(data)
            if img.shape[-1] == 1:
                img = np.repeat(img, 3, axis=-1)

            imgs_array.append(cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR))
    else:
        # load one compressed image after the other in RAM and uncompress
        for data in data_enum:
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
            imgs_array.append(img[:, :, [2, 1, 0]])  # from BGR to RGB

    return np.array(imgs_array)


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 0.0001
    image_writer_processes: int = 10
    image_writer_threads: int = 5
    video_backend: str | None = None


DEFAULT_DATASET_CONFIG = DatasetConfig()


class ConverterToLeRobotDataset():
    def __init__(self, repo_id: str, task, robot_type: str, custom_config, root: Path = HF_LEROBOT_HOME, mode: Literal["video", "image"] = "video", dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG, verbose=False):
        
        # Create info from custom config
        self.fps_used = custom_config["fps_used"][task]
        self.init_task = task

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
            if verbose:
                print(cam, (custom_config["cameras"][cam]["height"], custom_config["cameras"][cam]["width"], custom_config["cameras"][cam]["channels"]),)
            features[f"observation.images.{cam}"] = {
                "dtype": mode,
                "shape": (custom_config["cameras"][cam]["height"], custom_config["cameras"][cam]["width"], custom_config["cameras"][cam]["channels"]),
                "names": [
                    "height",
                    "width",
                    "channels",
                ],
            }

        if Path(root / repo_id).exists():
            shutil.rmtree(root / repo_id)
        
        self.dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=root / repo_id,
        fps=self.fps_used,
        robot_type=robot_type,
        features=features,
        use_videos=dataset_config.use_videos,
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
        video_backend=dataset_config.video_backend,
    )
        
        self.lerobot_path = root / repo_id

        self.custom_config = custom_config
        
    def populate(self, task, hdf5_path, episodes, show_data_analysis=False, verbose=False):
        if show_data_analysis and self.lerobot_path is not None:
            os.mkdir(self.lerobot_path / "data analysis")

        fps_used = self.custom_config["fps_used"][task]
        if np.abs(fps_used - self.fps_used) > 0.5:
            print(f"Warning : the fps {fps_used} used for task {task} is too different from the {self.fps_used} fps used for the initially given task {self.init_task}")
        
        for ep in tqdm.tqdm(episodes):

            (
                imgs_per_cam,
                state,
                action,
            ) = self.load_raw_episode_data(hdf5_path, ep, show_data_analysis, verbose)
            num_frames = state.shape[0]

            test_frames = {camera:[] for camera in imgs_per_cam.keys()}

            if verbose:
                print(f"adding {num_frames} frames")
            for i in range(num_frames):
                frame = {
                    "observation.state": state[i],
                    "action": action[i],
                }

                for camera, img_array in imgs_per_cam.items():
                    test_frames[camera].append(img_array[i])
                    frame[f"observation.images.{camera}"] = img_array[i]

                frame["task"] = task

                self.dataset.add_frame(frame)

            if verbose:
                self.dataset.save_episode()
            else:
                old_stdout, old_stderr, devnull = suppress_c_stdout_stderr()
                try:
                    self.dataset.save_episode()
                finally:
                    restore_c_stdout_stderr(old_stdout, old_stderr, devnull)

        return self.dataset
    
    def load_raw_episode_data(self,
    hdf5_path: Path,
    ep: int,
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
        
        action_ft_list = self.custom_config["action"]["hdf5_selected_names"]
        state_ft_list = self.custom_config["state"]["hdf5_selected_names"]
        outlier_deletion = self.custom_config["outlier_deletion"]

        with h5py.File(hdf5_path, "r") as file:
            # action
            action = []
            for action_ft_name in action_ft_list:
                action.append(torch.from_numpy(file[f"{ep:03d}/"+action_ft_name][:]))

            # state
            state = []
            for state_ft_name in state_ft_list:
                state.append(torch.from_numpy(file[f"{ep:03d}/"+state_ft_name][:]))

            # camera
            imgs_per_cam = self.load_raw_images_per_camera(
                file,
                ep,
                verbose,
            )

            # Data analysis
            if outlier_deletion or show_data_analysis:
                state_dists, img_dists, mean_dists = self.compute_data_analysis(state, imgs_per_cam, verbose)

                if verbose:
                    print("data analysis finished")

            # Data post-processing
            action = np.concatenate(action, 1)
            state = np.concatenate(state, 1)
            if outlier_deletion:
                out_indices = find_lower_outlier(mean_dists)

                action = np.concatenate([action[out_indices[i]+1:out_indices[i+1]] for i in range(len(out_indices)-1)]+[action[out_indices[-1]+1:]], axis=0)
                state = np.concatenate([state[out_indices[i]+1:out_indices[i+1]] for i in range(len(out_indices)-1)]+[state[out_indices[-1]+1:]], axis=0)
                for cam in imgs_per_cam.keys():
                    imgs_per_cam[cam] = np.concatenate([imgs_per_cam[cam][out_indices[i]+1:out_indices[i+1]] for i in range(len(out_indices)-1)]+[imgs_per_cam[cam][out_indices[-1]+1:]], axis=0)

                if verbose:
                    print("data processing finished")

            # Save data analysis
            if show_data_analysis and self.lerobot_path is not None:
                os.mkdir(self.lerobot_path / "data analysis" / str(ep))

                for state_ft_name in state_ft_list:
                    if outlier_deletion:
                        plt.scatter(out_indices, state_dists[state_ft_name][out_indices])
                    plt.plot(state_dists[state_ft_name])
                    plt.savefig(self.lerobot_path / "data analysis" / str(ep) / (state_ft_name.split("/")[-1]+".png"), bbox_inches="tight")
                    plt.close()

                    if verbose:
                        print("Data analysis plot saved as", self.lerobot_path / "data analysis" / str(ep) / (state_ft_name.split("/")[-1]+".png"))

                for cam in imgs_per_cam.keys():
                    if outlier_deletion:
                        plt.scatter(out_indices, img_dists[cam][out_indices])
                    plt.plot(img_dists[cam])
                    plt.savefig(self.lerobot_path / "data analysis" / str(ep) / (cam+".png"), bbox_inches="tight")
                    plt.close()

                    if verbose:
                        print("Data analysis plot saved as", self.lerobot_path / "data analysis" / str(ep) / (cam+".png"))

                labels = [key for key in state_dists.keys()]+[key for key in imgs_per_cam.keys()]
                x = np.concatenate([state_dists[key][:,None] for key in state_dists.keys()] + [img_dists[key][:,None] for key in imgs_per_cam.keys()], axis=1)

                self.plot_spearman_matrix(ep, x, labels)

                if outlier_deletion:
                    plt.scatter(out_indices, mean_dists[out_indices])
                plt.plot(mean_dists, label="total state distance")
                plt.legend(loc="best")
                plt.savefig(self.lerobot_path / "data analysis" / str(ep) / ("total_state.png"), bbox_inches="tight")
                plt.close()

        return (
            imgs_per_cam,
            state,
            action,
        )
    
    def compute_data_analysis(self, state, imgs_per_cam, verbose):
        state_ft_list = self.custom_config["state"]["hdf5_selected_names"]

        state_dists, img_dists = {}, {}
        for i, state_ft_name in enumerate(state_ft_list):
            state_dists[state_ft_name] = np.linalg.norm(np.diff(state[i], 1, axis=0), axis=-1)

        for cam in imgs_per_cam.keys():
            if verbose:
                print("data analysis of", imgs_per_cam[cam].shape[0], "elements for camera", cam)
            N = imgs_per_cam[cam].shape[0]
            img_dists[cam] = []
            for i in range(int(np.ceil(N/32))):
                imgcam = imgs_per_cam[cam][i*32:min((i+1)*32, N-1)+1].reshape(min((i+1)*32, N-1)+1 - i*32, -1)
                img_dists[cam].append(np.linalg.norm(np.diff(imgcam, 1, axis=0), axis=-1))

            img_dists[cam] = np.concatenate(img_dists[cam])

        if state_dists:
            mean_dists = np.mean([state_dists[key]/(1e-15+np.max(state_dists[key])) for key in state_dists.keys()], axis=0)
            if img_dists:
                mean_dists = np.mean([mean_dists, np.mean([img_dists[key]/(1e-15+np.max(img_dists[key])) for key in imgs_per_cam.keys()], axis=0)], axis=0)
        else:
            mean_dists = np.mean([img_dists[key]/(1e-15+np.max(img_dists[key])) for key in img_dists.keys()], axis=0)
        
        return state_dists, img_dists, mean_dists
    
    def plot_spearman_matrix(self, ep, x, labels):
        coef_mat = spearman_coefficient_matrix(x, x)

        xaxis = np.arange(len(labels))
        fig = plt.figure()
        ax = plt.gca()
        im = ax.matshow(coef_mat, interpolation='none')
        fig.colorbar(im)
        ax.set_xticks(xaxis)
        ax.set_yticks(xaxis)
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        fig.tight_layout()
        plt.savefig(self.lerobot_path / "data analysis" / str(ep) / ("spearman_matrix.png"), bbox_inches="tight")
        plt.close()

    def load_raw_images_per_camera(self, hd5_file: h5py.File, ep: int, verbose) -> dict[str, np.ndarray]:
        imgs_per_cam = {}
        camera_dict = self.custom_config["cameras"]
        if verbose:
            print("Processing images")
        for camera in camera_dict.keys():
            width, height, channel = camera_dict[camera]["width"], camera_dict[camera]["height"], camera_dict[camera]["channels"]
            if verbose:
                print("Processing camera", camera, " with (re)shape (", width, height, channel, ")")
            uncompressed = hd5_file[f"{ep:03d}/"+camera_dict[camera]["name"]].ndim == 4

            imgs_per_cam[camera] = process_image(hd5_file[f"{ep:03d}/"+camera_dict[camera]["name"]], width, height, uncompressed)
        if verbose:
            print("finished the processing")
        return imgs_per_cam


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

    converter = ConverterToLeRobotDataset(repo_id, task, "franka", config, mode=mode, dataset_config=dataset_config, verbose=verbose)

    converter.populate(task, hdf5_folder_path / hdf5_dataset_name, episodes, show_data_analysis=show_data_analysis, verbose=verbose)

    subprocess.run(["chmod", "-R", "777", HF_LEROBOT_HOME], check=True)

    if push_to_hub:
        converter.dataset.push_to_hub()


if __name__ == "__main__":
    tyro.cli(port_inria_franka)

    # example call:
    # $ python inria_tiago_hdf5_to_lerobot.py --hdf5_path place_fruit_in_bowl.h5 --repo_id /franka/place_fruit_in_bowl --task tiago_place_fruit_in_bowl

