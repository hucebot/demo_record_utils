# demo_record_utils

Utility functions for demo recording and post-processing.

- Folder `record` contains some example bash scripts to record demonstration via rosbag.
- Folder `postprocess` contains Python functions and example scripts to extract demonstration data from rosbags and create a HDF5 and a LeRobot dataset.

## Postprocess guide

Build docker image and run the container using the utility scripts `build_docker.sh` and `run_docker.sh`.

The `postprocess` folder will be mounted on the container and it is the container's start directory.
Please make sure to save the demonstration rosbags in a dedicated folder inside `postprocess` following the structure :

`<dataset-name>`
```
|___task1
    |___episode1
        |___teleop_result.db3
        |___metadata.yaml
    |___episode2
        |___teleop_result.db3
        |___metadata.yaml
    ...
|___task2
...
|___config.yaml
```

### rosbags -> HDF5

The script `inria_franka_rosbag_to_hdf5.py` converts manipulation demos (inside the folder `<dataset-name>`) into a HDF5 file.

```bash
python inria_franka_rosbag_to_hdf5.py --rosbag_folder <dataset-name> --hdf5_dir <dataset-name>_converted --tasks task1 task2 ... taskn
```

This will create an HDF5 file named `<task>.h5` for each task in $\{{\rm task1}, {\rm task2}, ..., {\rm taskn}\}$ and a default config file for possible conversion to a LeRobot dataset if there is none in a folder `<dataset-name>_converted` (created if it doesn't exist).

### HDF5 -> LeRobot

It is possible to create a LeRobot dataset based on an HDF5 dataset, using the script `inria_franka_hdf5_to_lerobot.py`.

```bash
python inria_franka_hdf5_to_lerobot.py --hdf5-folder-path <dataset-name>_converted --hdf5-dataset-name <dataset-name>.h5 --repo_id <org>/<dataset-name>_converted --task <task-name>
```

By default this script save the dataset locally (inside `HF_LEROBOT_HOME=/postprocess/lerobot`).
