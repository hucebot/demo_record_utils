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

Controls (assets/franka_buttons.json): home, tare FTS, record, cancel recording.

Robot configs (topics, demo name) are in `scripts/record.py:ROBOT_CONFIGS`.

### Headless (bash)

```bash
# Inside the container:
./record/record.sh --demo_task1        # start recording
./record/record.sh --clean demo_task1  # remove last bag
```

## Post-Processing Pipeline

```bash
make convert-hdf5 ROBOT=franka TASK=my_demo      # rosbags in /datasets/my_demo/ -> /datasets/hdf5_converted/my_demo.h5
pytest postprocess/                                # the conversion's tests (synthetic bags)
```

`postprocess/config_rosbag2hdf5/config.yaml` selects the topics and how frames are built (details in the docstring of
`postprocess/utils.py`):

- **Frames** are the reference topic's messages (a camera). Every other topic is sampled at them (the last message at
  or before the frame); frames before every topic has published once are dropped, so no frame uses data from its future.
- **Actions** (`action_time: next_frame`): the command in effect at the next frame, i.e. the command that followed the
  observation, which is what a policy must output. The pose command is `/cartesian_impedance/equilibrium_pose`, what
  teleoperation (and a policy at run time) sends to the controller, not the controller's filtered copy of it. `current` keeps the previous behaviour (the command the observation
  was already following, one frame late).
- **Times** (`time_source: header`): the messages' header stamps, the way ForceVAM's inference node lines its inputs
  up (the latest front_cam1 stamp, every other topic's last message at or before it), so training frames match what
  the policy sees at run time. Both assume the camera and robot computers share one clock; each demo stores the
  median receive-minus-stamp delay per topic (`median_header_delay_ms`) to check it.
- **Full-rate streams**: the numeric topics are also stored at their own rate under `data/demo_k/raw/` (e.g. the wrist
  force at 800 Hz, which sampling at 30 Hz would alias), with their times.
- **Bookkeeping**: bags are converted in name order (= recording order) and each demo stores its `bag_name`;
  converting again only adds the new bags. A bag missing a topic is skipped (listed at the end), the others are
  converted. Topics marked `optional: true` may be absent (e.g. the raw wrench in bags recorded before it was added).

While recording, the Stream Deck checks each saved bag: every topic must have messages (counts and rates are logged),
otherwise the RECORD key shows CHECK BAG. CANCEL RECORD deletes the recording in progress, or, when idle, the last
saved demo.
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
