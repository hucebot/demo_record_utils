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
    task,
    robot_type: str,
    root: Path = HF_LEROBOT_HOME,
    mode: Literal["video", "image"] = "video",
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
    custom_config=None,
    verbose=False,
) -> LeRobotDataset:
    fps_used = custom_config["fps_used"][task]

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

    if verbose:
        print(repo_id, "stop",
            root / repo_id, "stop",
            fps_used, "stop",
            robot_type, "stop",
            features, "stop",
            dataset_config.use_videos, "stop",
            dataset_config.tolerance_s, "stop",
            dataset_config.image_writer_processes, "stop",
            dataset_config.image_writer_threads, "stop",
            dataset_config.video_backend, "stop")

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
    print("Processing images")
    for camera in camera_dict.keys():
        width, height, channel = camera_dict[camera]["width"], camera_dict[camera]["height"], camera_dict[camera]["channels"]
        print("Processing camera", camera, " with (re)shape (", width, height, channel, ")")
        uncompressed = hd5_file[f"{ep:03d}/"+camera_dict[camera]["name"]].ndim == 4

        if uncompressed:
            # load all images in RAM
            imgs_array = []
            for data in hd5_file[f"{ep:03d}/"+camera_dict[camera]["name"]]:
                img = data if data.shape[-1] == 3 else cv2.cvtColor(data[:,:,None], cv2.COLOR_GRAY2BGR)
                img = cv2.normalize(img, None, 0.0, 1.0, cv2.NORM_MINMAX).astype(np.float32)
                #print(img)
                imgs_array.append(cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR))
            imgs_array = np.array(imgs_array)
        else:
            # load one compressed image after the other in RAM and uncompress
            imgs_array = []
            for data in hd5_file[f"{ep:03d}/"+camera_dict[camera]["name"]]:
                img = cv2.imdecode(data, cv2.IMREAD_COLOR)
                img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
                imgs_array.append(img[:, :, [2, 1, 0]])  # from BGR to RGB
            imgs_array = np.array(imgs_array)

        imgs_per_cam[camera] = imgs_array
    print("finished the processing")
    return imgs_per_cam


def load_raw_episode_data(
    hdf5_path: Path,
    ep: int,
    action_ft_list,
    state_ft_list,
    camera_dict,
    show_data_analysis,
    lerobot_path,
    outlier_deletion,
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

        #for key in all_keys:
            #split_key = key.split("/")
            #if len(split_key) < 3:
            #    continue
            
            #print(key)
            #print(file[key][:].shape)

        #print(file["000/observations/f_ext"][:])

        # action
        action = []
        for action_ft_name in action_ft_list:
            action.append(torch.from_numpy(file[f"{ep:03d}/"+action_ft_name][:]))

        # state
        state = []
        for state_ft_name in state_ft_list:
            state.append(torch.from_numpy(file[f"{ep:03d}/"+state_ft_name][:]))

        # camera
        imgs_per_cam = load_raw_images_per_camera(
            file,
            ep,
            camera_dict,
        )

        # Data analysis
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
                #print(i, imgcam.shape)
                img_dists[cam].append(np.linalg.norm(np.diff(imgcam, 1, axis=0), axis=-1))
                #print(img_dists[cam])

            img_dists[cam] = np.concatenate(img_dists[cam])
            #print(img_dists[cam].shape, N)

        if state_dists:
            mean_dists = np.mean([state_dists[key]/(1e-15+np.max(state_dists[key])) for key in state_dists.keys()], axis=0)
            if img_dists:
                print(mean_dists.shape)
                mean_dists = np.mean([mean_dists, np.mean([img_dists[key]/(1e-15+np.max(img_dists[key])) for key in imgs_per_cam.keys()], axis=0)], axis=0)
        else:
            mean_dists = np.mean([img_dists[key]/(1e-15+np.max(img_dists[key])) for key in img_dists.keys()], axis=0)

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

        print("data processing finished")
        # Save data analysis
        if show_data_analysis and lerobot_path is not None:
            os.mkdir(lerobot_path / "data analysis" / str(ep))

            for state_ft_name in state_ft_list:
                if outlier_deletion:
                    plt.scatter(out_indices, state_dists[state_ft_name][out_indices])
                plt.plot(state_dists[state_ft_name])
                plt.savefig(lerobot_path / "data analysis" / str(ep) / (state_ft_name.split("/")[-1]+".png"), bbox_inches="tight")
                plt.close()

                if verbose:
                    print("Data analysis plot saved as", lerobot_path / "data analysis" / str(ep) / (state_ft_name.split("/")[-1]+".png"))

            for cam in imgs_per_cam.keys():
                if outlier_deletion:
                    plt.scatter(out_indices, img_dists[cam][out_indices])
                plt.plot(img_dists[cam])
                plt.savefig(lerobot_path / "data analysis" / str(ep) / (cam+".png"), bbox_inches="tight")
                plt.close()

                if verbose:
                    print("Data analysis plot saved as", lerobot_path / "data analysis" / str(ep) / (cam+".png"))

            labels = [key for key in state_dists.keys()]+[key for key in imgs_per_cam.keys()]
            x = np.concatenate([state_dists[key][:,None] for key in state_dists.keys()] + [img_dists[key][:,None] for key in imgs_per_cam.keys()], axis=1)
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
            plt.savefig(lerobot_path / "data analysis" / str(ep) / ("spearman_matrix.png"), bbox_inches="tight")
            plt.close()

            if outlier_deletion:
                plt.scatter(out_indices, mean_dists[out_indices])
            plt.plot(mean_dists, label="total state distance")
            plt.legend(loc="best")
            plt.savefig(lerobot_path / "data analysis" / str(ep) / ("total_state.png"), bbox_inches="tight")
            plt.close()

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
    lerobot_path=None,
    verbose=False,
) -> LeRobotDataset:
    
    if show_data_analysis and lerobot_path is not None:
        os.mkdir(lerobot_path / "data analysis")
    
    for ep in tqdm.tqdm(episodes):

        (
            imgs_per_cam,
            state,
            action,
        ) = load_raw_episode_data(hdf5_path, ep, custom_config["action"]["hdf5_selected_names"], custom_config["state"]["hdf5_selected_names"], custom_config["cameras"], show_data_analysis, lerobot_path, custom_config["outlier_deletion"], verbose)
        num_frames = state.shape[0]

        test_frames = {camera:[] for camera in imgs_per_cam.keys()}

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
    
    #imgs_per_cam = load_raw_images_per_camera(f, episodes[0], config["cameras"])
    #cam_resolutions = {cam:imgs_per_cam[cam].shape[1:] for cam in imgs_per_cam.keys()}
    #del imgs_per_cam

    dataset = create_empty_dataset(
        repo_id,
        task,
        robot_type="franka",
        mode=mode,
        dataset_config=dataset_config,
        custom_config=config,
        verbose=verbose,
    )
    dataset = populate_dataset(
        dataset,
        hdf5_folder_path / hdf5_dataset_name,
        task=task,
        episodes=episodes,
        custom_config=config,
        show_data_analysis=show_data_analysis,
        lerobot_path=HF_LEROBOT_HOME / repo_id,
        verbose=verbose,
    )

    subprocess.run(["chmod", "-R", "777", HF_LEROBOT_HOME], check=True)

    if push_to_hub:
        dataset.push_to_hub()


if __name__ == "__main__":
    tyro.cli(port_inria_franka)

    # example call:
    # $ python inria_tiago_hdf5_to_lerobot.py --hdf5_path place_fruit_in_bowl.h5 --repo_id /franka/place_fruit_in_bowl --task tiago_place_fruit_in_bowl

