# demo_record_utils

Utility functions for demo recording and post-processing.

- Folder `record` contains some example bash scripts to record demonstration via rosbag.
- Folder `postprocess` contains Python functions and example scripts to extract demonstration data from rosbags and create a HDF5 and a LeRobot dataset.

## Postprocess guide

Build docker image and run the container using the utility scripts `build_docker.sh` and `run_docker.sh`.

The `postprocess` folder will be mounted on the container and it is the container's start directory.
Please make sure to save the demonstration rosbags in a dedicated folder inside `postprocess` following the structure :

`<dataset-name>`
|___data
    |___episode1.db3
    |___episode2.db3
    ...
|___config.yaml
|___metadata.yaml

### rosbags -> HDF5

The script `inria_franka_rosbag_to_hdf5.py` converts manipulation demos (inside the folder `<dataset-name>`) into a HDF5 file.

```bash
python inria_franka_rosbag_to_hdf5.py --rosbag_folder <dataset-name> --hdf5_dir <dataset-name>_converted
```

This will create a folder `<dataset-name>_converted` containing an HDF5 file named `<dataset-name>.h5` and a default config file for possible conversion to a LeRobot dataset.

### HDF5 -> LeRobot

It is possible to create a LeRobot dataset based on an HDF5 dataset, using the script `inria_franka_hdf5_to_lerobot.py`.

```bash
python inria_franka_hdf5_to_lerobot.py --hdf5-folder-path <dataset-name>_converted --hdf5-dataset-name <dataset-name>.h5 --repo_id <org>/<dataset-name>_converted --task <task-name>
```

By default this script save the dataset locally (inside `HF_LEROBOT_HOME=/postprocess/lerobot`).
