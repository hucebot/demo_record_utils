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

  The configuration file in `postprocess/rosbag_dataset` (resp. `postprocess/converted`) gives the core logic for the **Rosbag to HDF5** (resp. **HDF5 to LeRobot**) conversion.

---

### ⚙️ Rosbag to HDF5 Configuration

To handle the asynchronous nature of ROS2 (where different sensors publish at different rates), the conversion script uses a **Reference-Based Sampling** strategy.

#### 🕒 Temporal Synchronization
The conversion is governed by the `reference_topic_name`. 

* **The Master Clock:** The script iterates through every message in the **reference topic** (e.g., the depth camera).
* **Zero-Order Hold Sampling:** For every timestamp recorded by the reference topic, the script looks back and grabs the **most recent message** from every other selected topic.
* **The Result:** A perfectly synchronized dataset where every "frame" contains a matching set of images, robot states, and actions, even if they originally arrived at different times.

#### 🛠️ Key Features
* **Physical Quantity Filtering:** Automatically extracts only the relevant array (e.g., just the `position` vector) from complex ROS messages like `sensor_msgs/JointState`.
* **Automatic HDF5 Structuring:** Groups data into `observations/` and `actions/` hierarchies, making it natively compatible with the next step in the LeRobot pipeline.

To wrap up the pipeline, here is the professional description for the **HDF5 to LeRobot** configuration. This stage focuses on dataset "distillation"—cleaning the data and formatting it for high-performance training.

---

### 🏗️ HDF5 to LeRobot Configuration

This configuration defines how the intermediate HDF5 data is refined and packaged into a **LeRobot** dataset. It handles the final mapping of tensors and applies data-cleaning heuristics to improve training quality.

#### 🧹 Data Distillation & Cleaning
Beyond simple formatting, this stage applies two critical processing steps:

* **Outlier Deletion**: When enabled, the script employs an algorithm to prune "static" samples. If the state-action pair remains nearly identical over consecutive frames (i.e., the robot and the environment are idling), those samples are removed. This prevents the model from being biased toward stationary behavior and reduces dataset redundancy.
* **Video Encoding (`fps_used`)**: Defines the frames-per-second used when encoding the raw image buffers into compressed video streams. This ensures that the temporal playback during training matches the real-world physics of the demonstration.



#### 🗺️ Tensor Mapping Logic
The configuration maps HDF5 paths to LeRobot-standardized naming conventions:

| Feature | HDF5 Source | LeRobot Destination | Description |
| :--- | :--- | :--- | :--- |
| **Actions** | `actions/desired_ee_pos`, etc. | `motor` | Flattened vector of all control commands. |
| **State** | `observations/joint_pos`, etc. | `observation.state` | Unified proprioceptive feedback vector. |
| **Visuals** | `observations/front_cam1`, etc. | `observation.images` | Reference to the encoded video streams. |

#### 🛠️ Key Features
* **Feature Flattening**: Automatically converts multi-source HDF5 groups (like 7 joint positions + 1 gripper width) into a single, flat vector labeled for the LeRobot learner (e.g., `joint_pos_0` through `joint_pos_6`).
* **Dimension Specification**: Explicitly declares camera resolutions and channels to ensure the training dataloader allocates the correct tensor shapes ($C \times H \times W$).