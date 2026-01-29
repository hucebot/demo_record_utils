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

def find_lower_outlier(x, verbose=False):
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

        if verbose:
            print(new_last_dec_idx, len(Femp)//2, "gamma : ", gamma, "last idx : ", last_dec_idx)

        val_flag = (last_dec_idx[1] - last_dec_idx[0]) / (gamma[1] - gamma[0])
        if val_flag > 40000:
            break

    return np.sort(sorted_idx[:last_dec_idx[0]+1])

def process_image(data, width, height, uncompressed):
    if uncompressed:
        img = data/np.max(data)
        if img.shape[-1] == 1:
            img = np.repeat(img, 3, axis=-1)

        return cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
    else:
        # load one compressed image after the other in RAM and uncompress
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
        return img[:, :, [2, 1, 0]]  # from BGR to RGB


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 0.0001
    image_writer_processes: int = 10
    image_writer_threads: int = 5
    video_backend: str | None = None


DEFAULT_DATASET_CONFIG = DatasetConfig()


class ConverterToLeRobotDataset():
    def __init__(self, repo_id: str, task, robot_type: str, custom_config, root: Path = HF_LEROBOT_HOME, mode: Literal["video", "image"] = "video", dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG, verbose=False, loading_batch_size=32):
        self.verbose = verbose

        # Create info from custom config
        self.loading_batch_size = loading_batch_size
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

        while Path(root / repo_id).exists():
            repo_id += "_twin"
        
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
        
    def populate(self, task, hdf5_path, episodes, show_data_analysis=False):
        if show_data_analysis and self.lerobot_path is not None:
            os.mkdir(self.lerobot_path / "data analysis")

        fps_used = self.custom_config["fps_used"][task]
        if np.abs(fps_used - self.fps_used) > 0.5:
            print(f"Warning : the fps {fps_used} used for task {task} is too different from the {self.fps_used} fps used for the initially given task {self.init_task}")
        
        outlier_deletion = self.custom_config["outlier_deletion"]
        camera_keys = self.custom_config["cameras"].keys()
        state_ft_list = self.custom_config["state"]["hdf5_selected_names"]

        with h5py.File(hdf5_path, "r") as file:
            for ep in tqdm.tqdm(episodes):
                    
                    if show_data_analysis or outlier_deletion:
                        self.state_dists, self.img_dists, self.mean_dists = {key:[] for key in state_ft_list}, {key:[] for key in camera_keys}, []
                        self.last_state, self.last_img_per_cam = None, {}

                    num_frames = file[f"{ep:03d}/"+self.custom_config["state"]["hdf5_selected_names"][-1]][:].shape[0]

                    if self.verbose:
                        print(f"adding {num_frames} frames")

                    num_iters = num_frames//self.loading_batch_size
                    for loading_batch_idx in range(num_iters):
                        self.add_batch_raw_episode_data(loading_batch_idx*self.loading_batch_size, (loading_batch_idx+1)*self.loading_batch_size, file, task, ep, show_data_analysis)

                    self.add_batch_raw_episode_data(num_iters*self.loading_batch_size, num_frames, file, task, ep, show_data_analysis)

                    if show_data_analysis or outlier_deletion:
                        self.compute_mean_dists()
                    
                    if outlier_deletion:
                        out_indices = find_lower_outlier(self.mean_dists, self.verbose)
                        for loading_batch_idx in range(num_iters):
                            self.add_batch_raw_episode_data(loading_batch_idx*self.loading_batch_size, (loading_batch_idx+1)*self.loading_batch_size, file, task, ep, show_data_analysis, out_indices=out_indices)

                        self.add_batch_raw_episode_data(num_iters*self.loading_batch_size, num_frames, file, task, ep, show_data_analysis, out_indices=out_indices)

                        if show_data_analysis:
                            self.save_data_analysis(ep, out_indices)
                    elif show_data_analysis:
                        self.save_data_analysis(ep, None)

                    if self.verbose:
                        self.dataset.save_episode()
                    else:
                        old_stdout, old_stderr, devnull = suppress_c_stdout_stderr()
                        try:
                            self.dataset.save_episode()
                        finally:
                            restore_c_stdout_stderr(old_stdout, old_stderr, devnull)

        return self.dataset
    
    def add_batch_raw_episode_data(self,
                              start,
                              end,
                              file,
                              task,
                              ep,
                              show_data_analysis,
                              out_indices=None,
):
        
        action_ft_list = self.custom_config["action"]["hdf5_selected_names"]
        state_ft_list = self.custom_config["state"]["hdf5_selected_names"]
        outlier_deletion = self.custom_config["outlier_deletion"]

        # action
        action = []
        for action_ft_name in action_ft_list:
            action.append(torch.from_numpy(file[f"{ep:03d}/"+action_ft_name][:])[start:end])

        # state
        state = []
        for state_ft_name in state_ft_list:
            state.append(torch.from_numpy(file[f"{ep:03d}/"+state_ft_name][:])[start:end])

        # camera
        imgs_per_cam = self.load_raw_images_per_camera(start, end, file, ep)

        # Data analysis
        # If we are trying to detect the outliers, we only need to compute the stats
        if outlier_deletion and (out_indices is None):
            self.compute_state_and_img_dists(state, imgs_per_cam)

            if self.verbose:
                print("data analysis finished for batch", start, ":", end)
        # Otherwise, we build the actual converted dataset
        else:
            # If outlier_deletion is False but we still want data analysis, we need to compute the stats
            if show_data_analysis and (out_indices is None):
                self.compute_state_and_img_dists(state, imgs_per_cam)

                if self.verbose:
                    print("data analysis finished for batch", start, ":", end)

            # Data post-processing
            action = np.concatenate(action, 1)
            state = np.concatenate(state, 1)
            
            if outlier_deletion:
                action = np.concatenate([action[out_indices[i]+1:out_indices[i+1]] for i in range(len(out_indices)-1)]+[action[out_indices[-1]+1:]], axis=0)
                state = np.concatenate([state[out_indices[i]+1:out_indices[i+1]] for i in range(len(out_indices)-1)]+[state[out_indices[-1]+1:]], axis=0)
                for cam in imgs_per_cam.keys():
                    imgs_per_cam[cam] = np.concatenate([imgs_per_cam[cam][out_indices[i]+1:out_indices[i+1]] for i in range(len(out_indices)-1)]+[imgs_per_cam[cam][out_indices[-1]+1:]], axis=0)

                if self.verbose:
                    print("data processing finished for batch", start, ":", end)

            for i in range(state.shape[0]):
                frame = {
                    "observation.state": state[i],
                    "action": action[i],
                }

                for camera, img_array in imgs_per_cam.items():
                    frame[f"observation.images.{camera}"] = img_array[i]

                frame["task"] = task

                self.dataset.add_frame(frame)
    
    def compute_state_and_img_dists(self, state, imgs_per_cam):
        state_ft_list = self.custom_config["state"]["hdf5_selected_names"]

        for i, state_ft_name in enumerate(state_ft_list):
            new_state = state[i]
            if self.last_state is not None:
                new_state = np.concatenate((self.last_state[i], state[i]), axis=0)
                
            self.state_dists[state_ft_name].append(np.linalg.norm(np.diff(new_state, 1, axis=0), axis=-1))

        self.last_state = [stt[None, -1] for stt in state]

        for cam in imgs_per_cam.keys():
            if self.verbose:
                print("data analysis of", imgs_per_cam[cam].shape[0], "elements for camera", cam)
            N = imgs_per_cam[cam].shape[0]
            imgcam = imgs_per_cam[cam].reshape(N, -1)
            if cam in self.last_img_per_cam.keys():
                imgcam = np.concatenate((self.last_img_per_cam[cam], imgcam), axis=0)

            self.img_dists[cam].append(np.linalg.norm(np.diff(imgcam, 1, axis=0), axis=-1))

            self.last_img_per_cam[cam] = imgcam[None, -1]

    def compute_mean_dists(self):
        if self.state_dists:
            for key in self.state_dists.keys():
                self.state_dists[key] = np.concatenate(self.state_dists[key], axis=0)

            self.mean_dists = np.mean([self.state_dists[key]/(1e-15+np.max(self.state_dists[key])) for key in self.state_dists.keys()], axis=0)
            if self.img_dists:
                for key in self.img_dists.keys():
                    self.img_dists[key] = np.concatenate(self.img_dists[key], axis=0)

                self.mean_dists = np.mean([self.mean_dists, np.mean([self.img_dists[key]/(1e-15+np.max(self.img_dists[key])) for key in self.img_dists.keys()], axis=0)], axis=0)
        else:
            for key in self.img_dists.keys():
                self.img_dists[key] = np.concatenate(self.img_dists[key], axis=0)

            self.mean_dists = np.mean([self.img_dists[key]/(1e-15+np.max(self.img_dists[key])) for key in self.img_dists.keys()], axis=0)
    
    def save_data_analysis(self, ep, out_indices):
        outlier_deletion = self.custom_config["outlier_deletion"]
        os.mkdir(self.lerobot_path / "data analysis" / str(ep))

        for state_ft_name in self.state_dists.keys():
            if self.verbose:
                print("state dist shape :", self.state_dists[state_ft_name].shape)
            if outlier_deletion:
                plt.scatter(out_indices, self.state_dists[state_ft_name][out_indices])
            plt.plot(self.state_dists[state_ft_name])
            plt.savefig(self.lerobot_path / "data analysis" / str(ep) / (state_ft_name.split("/")[-1]+".png"), bbox_inches="tight")
            plt.close()

            if self.verbose:
                print("Data analysis plot saved as", self.lerobot_path / "data analysis" / str(ep) / (state_ft_name.split("/")[-1]+".png"))

        for cam in self.img_dists.keys():
            if outlier_deletion:
                plt.scatter(out_indices, self.img_dists[cam][out_indices])
            plt.plot(self.img_dists[cam])
            plt.savefig(self.lerobot_path / "data analysis" / str(ep) / (cam+".png"), bbox_inches="tight")
            plt.close()

            if self.verbose:
                print("Data analysis plot saved as", self.lerobot_path / "data analysis" / str(ep) / (cam+".png"))

        labels = [key for key in self.state_dists.keys()]+[key for key in self.img_dists.keys()]
        x = np.concatenate([self.state_dists[key][:,None] for key in self.state_dists.keys()] + [self.img_dists[key][:,None] for key in self.img_dists.keys()], axis=1)

        self.plot_spearman_matrix(ep, x, labels)

        if outlier_deletion:
            plt.scatter(out_indices, self.mean_dists[out_indices])
        plt.plot(self.mean_dists, label="total state distance")
        plt.legend(loc="best")
        plt.savefig(self.lerobot_path / "data analysis" / str(ep) / ("total_state.png"), bbox_inches="tight")
        plt.close()
    
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

    def load_raw_images_per_camera(self, start, end, hd5_file: h5py.File, ep: int) -> dict[str, np.ndarray]:
        imgs_per_cam = {}
        camera_dict = self.custom_config["cameras"]
        if self.verbose:
            print("Processing images")
        for camera in camera_dict.keys():
            width, height, channel = camera_dict[camera]["width"], camera_dict[camera]["height"], camera_dict[camera]["channels"]
            if self.verbose:
                print("Processing camera", camera, " with (re)shape (", width, height, channel, ")")
            uncompressed = hd5_file[f"{ep:03d}/"+camera_dict[camera]["name"]].ndim == 4
            imgs_per_cam[camera] = []
            for img in hd5_file[f"{ep:03d}/"+camera_dict[camera]["name"]][start:end]:
                imgs_per_cam[camera].append(process_image(img, width, height, uncompressed))
            imgs_per_cam[camera] = np.array(imgs_per_cam[camera])

        if self.verbose:
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
    loading_batch_size=1024,
):
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

    converter = ConverterToLeRobotDataset(repo_id, task, "franka", config, mode=mode, dataset_config=dataset_config, verbose=verbose, loading_batch_size=loading_batch_size)

    converter.populate(task, hdf5_folder_path / hdf5_dataset_name, episodes, show_data_analysis=show_data_analysis)

    subprocess.run(["chmod", "-R", "777", HF_LEROBOT_HOME / repo_id], check=True)

    if push_to_hub:
        converter.dataset.push_to_hub()


if __name__ == "__main__":
    tyro.cli(port_inria_franka)

    # example call:
    # $ python inria_tiago_hdf5_to_lerobot.py --hdf5_path place_fruit_in_bowl.h5 --repo_id /franka/place_fruit_in_bowl --task tiago_place_fruit_in_bowl

