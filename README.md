# Demo Record Utils

Record robotic demonstrations via ROS2 and post-process them into HDF5 / LeRobot datasets.

## Quickstart

All operations are driven through `make`. The default robot is `franka`; override with `ROBOT=tiago`.

```bash
make build        # docker compose build
make run          # docker compose up -d (X11 + GPU passthrough)
make down         # docker compose down
make shell        # exec into container
make logs         # tail container logs
```

## Recording

### Interactive (Stream Deck)

```bash
make record ROBOT=franka TASK=my_demo
# or with a custom task name:
make record ROBOT=franka TASK=demo_task1
```

This runs `scripts/record.py` inside the container. Stream Deck button layouts are in `assets/franka_buttons.json` / `assets/tiago_buttons.json`.

Controls: home, resume, soft estop, gripper (open/close), tare FTS, record/cancel recording.

Robot configs (topics, demo name) are in `scripts/record.py:ROBOT_CONFIGS`.

### Headless (bash)

```bash
# Inside the container:
./record/record.sh --demo_task1        # start recording
./record/record.sh --clean demo_task1  # remove last bag
```

## Post-Processing Pipeline

WIP
## Make Rules Reference

| Rule       | Description                                      |
|------------|--------------------------------------------------|
| `build`    | Build the Docker image                           |
| `run`      | Start the container (detached)                   |
| `down`     | Stop the container                               |
| `shell`    | Open a bash shell inside the container           |
| `logs`     | Tail container logs                              |
| `record`   | Launch interactive recording (`ROBOT`, `TASK`)   |

Variables: `ROBOT=franka|tiago` (default: `franka`), `TASK=<name>` (default: `test_task`).

## Project Layout

```
.ci/Dockerfile            # ROS2 Humble + FFmpeg (SVT-AV1) + LeRobot + StreamDeck
docker-compose.yml        # data_collector service with GPU / host networking
Makefile                  # All build/run/record commands

record/record.sh          # headless rosbag recording
scripts/record.py         # Stream Deck ROS2 node (driven by make record)

postprocess/
  inria_franka_rosbag_to_hdf5.py   # rosbag → HDF5
  inria_franka_hdf5_to_lerobot.py  # HDF5 → LeRobot
  utils.py                          # extraction / sync utilities
  config_rosbag2hdf5/config.yaml    # topic selection for extraction
  config_hdf52lerobot/config.yaml   # tensor mapping for LeRobot

assets/
  franka_buttons.json   # Stream Deck layout for Franka
  tiago_buttons.json    # Stream Deck layout for Tiago
```
