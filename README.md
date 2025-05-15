# demo_record_utils

Utility functions for demo recording and post-processing.

- Folder `record` contains some example bash scripts to record demonstration via rosbag.
- Folder `postprocess` contains Python functions and example scripts to extract demonstration data from rosbags and create a HDF5 and a LeRobot dataset.

## Postprocess guide

Build docker image and run the container using the utility scripts `build_docker.sh` and `run_docker.sh`.

The `postprocess` folder will be mounted on the container and it is the container's start directory.
Please make sure to save the demonstration rosbags in a dedicated folder inside `postprocess`.

### rosbags -> HDF5

The script `manip_demo_rosbag_to_hdf5.py` is an example to convert manipulation demos (inside the folder `<demo-name>`) into a HDF5 file.

```bash
python manip_demo_rosbag_to_hdf5.py --folder <demo-name>
```

This will create a HDF5 file named `<demo-name>.h5`.

### HDF5 -> LeRobot

This can be used to create a LeRobot dataset, using the script `inria_tiago_hdf5_to_lerobot.py`.

```bash
python inria_tiago_hdf5_to_lerobot.py --hdf5_path <demo-name>.h5 --repo_id <org>/<dataset-name> --task <task-name>
```

By default this script save the dataset locally (inside `HF_LEROBOT_HOME=/postprocess/lerobot`).

## Use of LeRobot So100
We provide a straightforward way to use LeRobot So100 with the LeRobot library. The instructions and how to compile the docker are in the folder `docker_so100`.

python manip_demo_rosbag_to_hdf5.py --folder /rosbags/picking_mustard_3_places
python inria_tiago_hdf5_to_lerobot.py --hdf5_path /rosbags/picking_mustard_3_places.h5 --repo_id inria/picking_mustard_3_places --task picking_mustard_3_places