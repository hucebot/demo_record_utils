# 🛠️ Demo Record Utils

[](https://www.python.org/)
[](https://docs.ros.org/)
[](https://github.com/huggingface/lerobot)

A streamlined utility suite for **recording robotic demonstrations** via ROS2 and **post-processing** them into high-performance formats like HDF5 and LeRobot datasets.

-----

## 📂 Project Structure

  * **`/record`**: Essential bash scripts for automated demonstration recording via `rosbag`.
  * **`/postprocess`**: Python-based pipeline to extract, clean, and convert bag files into machine-learning-ready datasets.

-----

## 🚀 Getting Started

### 1\. Environment Setup

We provide a Dockerized environment to ensure all dependencies (ROS2, HDF5, LeRobot) are handled seamlessly.

```bash
# Build the utility image
./build_docker.sh

# Spin up the container
./run_docker.sh
```

*Note: The `/postprocess` folder is automatically mounted as the working directory inside the container.*

### 2\. Data Organization

For the pipeline to function correctly, organize your raw rosbags using the following directory tree:

```text
📂 <dataset-name>
├── 📂 task_01
│   ├── 📂 episode_01
│   │   ├── metadata.yaml
│   │   └── teleop_result_0.db3
│   │   └── teleop_result_1.db3
│   │   ...
│   └── 📂 episode_02
│       └── ...
├── 📂 task_02
│   └── ...
└── config.yaml
```

-----

## 🔄 Post-Processing Pipeline (Basic Usage)

### Phase I: Rosbags ⮕ HDF5

Convert raw manipulation demos into a unified HDF5 format. This step handles data extraction based on your specific `config.yaml`.

```bash
python inria_franka_rosbag_to_hdf5.py \
    --rosbag_folder <dataset-name> \
    --hdf5_dir <dataset-name>_converted \
    --tasks task1 task2
```

> **Output:** Creates `<task>.h5` files and generates a default configuration for the next phase.

### Phase II: HDF5 ⮕ LeRobot

Prepare your data for training by converting HDF5 files into the [LeRobot](https://github.com/huggingface/lerobot) format.

```bash
python inria_franka_hdf5_to_lerobot.py \
    --hdf5-folder-path <dataset-name>_converted \
    --hdf5-dataset-name <dataset-name>.h5 \
    --repo_id <org>/<dataset-name>_converted \
    --task <task-name>
```

**Default Local Path:** `/postprocess/lerobot` (configured via `HF_LEROBOT_HOME`).

-----

## 🛠️ Configuration

Customization is handled via YAML. You can find templates for these in:

  * `postprocess/rosbag_dataset`: For Rosbag extraction.
  * `postprocess/converted`: For LeRobot dataset metadata.
