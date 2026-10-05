"""
Utility functions to convert recorded rosbags (one folder per demo) into one HDF5 file per task.

Layout written (read by inria_franka_hdf5_to_lerobot.py and ForceVAM):
    data/demo_<k>/timestamps                (T, 1) float32, s from the first kept frame
    data/demo_<k>/actions                   (T, A) float32, the "actions/..." topics concatenated (attr action_parts)
    data/demo_<k>/obs/<name>                (T, ...) the "observations/..." topics, sampled at the reference frames
    data/demo_<k>/raw/<field>, <field>_t    every message of each numeric topic at its own rate (e.g. the wrist force
                                            at 800 Hz, which the frame sampling would alias), times in s on the same clock;
                                            <field> is the config's hdf5_name (raw/observations/ee_force, ...)
    attrs: bag_name (the demo's rosbag folder), num_samples, action_time, time_source, ...

How a frame is built (config.yaml, all optional):
    reference_topic_name   the frames are this topic's messages (a camera)
    action_time            "next_frame" (default): the action of frame k is the command in effect at frame k+1, i.e.
                           the command that followed the observation (what a policy must output); the last frame,
                           which has no next frame, is dropped. "current": the command in effect at frame k (the
                           previous behaviour: the command the observation was already following).
    time_source            "receive" (default): the time the recorder received each message (one clock for every
                           topic); "header": the message's header stamp (the time it was measured; only if every
                           publisher's clock is the same), falling back to the receive time where the stamp is 0
    image_size             [width, height] every camera is resized to (default [256, 256])
    save_raw_streams       true (default): also store the numeric topics at full rate under raw/
Observations and actions are the last message at or before the frame time (zero-order hold); frames before every
selected topic has published once are dropped (no frame ever uses a message from its future).

Demos are converted in the order of their folder names (the recorder names them <task>_<YYYYmmdd_HHMMSS>, so in
recording order) and recognised by bag_name: converting again adds only the new bags, as the next demo_<k>. A demo
is written under a temporary name and renamed when complete, so an interrupted conversion leaves nothing half-written
behind. A bag missing a selected topic, or with one that never published, is skipped with an error and listed at the
end; the others are converted.
"""

import time

import cv2
import h5py
import numpy as np
import yaml
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore, get_types_from_msg

from messages import IMAGE_TYPES, NUMERIC_TYPES, image_array, numeric_values, raw_image_to_array

# Your custom message definition
# check: https://ternaris.gitlab.io/rosbags/examples/register_types.html#from-multiple-files
GRIPPER_WIDTH_MSG = """
std_msgs/Header header
float32 width
"""

TMP_PREFIX = "_incomplete_"


# colors for printing
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def message_time(msg, receive_ns, time_source):
    """ms, from the header stamp ("header", where set) or the recorder's receive time."""
    if time_source == "header" and hasattr(msg, "header"):
        stamp = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
        if stamp > 0:
            return stamp * 1e-6
    return receive_ns * 1e-6


def extract_all_topics_single_pass(bagpaths, selected_topics, topic_types, time_source="receive",
                                   image_size=(256, 256), verbose=False):
    """
    Every message of the selected topics, in one pass over the bag files. Returns per topic: times (n,) in ms
    (float64), and per HDF5 field its values (n, ...); images decoded to RGB and resized to image_size (width, height).
    Also per topic the median header-minus-receive delay in ms (where the messages have a header), for the record.
    """
    if verbose:
        print(f"Extracting {len(selected_topics)} topics from {len(bagpaths)} bag file(s)")
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    typestore.register(get_types_from_msg(GRIPPER_WIDTH_MSG, 'custom_msgs/msg/GripperWidth'))
    for topic in selected_topics:
        if topic_types[topic] not in IMAGE_TYPES + NUMERIC_TYPES:
            raise NotImplementedError(f"{topic}: message type {topic_types[topic]} is not handled")

    times = {t: [] for t in selected_topics}
    delays = {t: [] for t in selected_topics}
    values = {t: {h: [] for h in m} for t, m in selected_topics.items()}
    for bagpath in bagpaths:
        with Reader(bagpath) as reader:
            connections = [x for x in reader.connections if x.topic in selected_topics]
            for connection, timestamp, rawdata in reader.messages(connections=connections):
                topic = connection.topic
                msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
                t = message_time(msg, timestamp, time_source)
                times[topic].append(t)
                if hasattr(msg, "header") and msg.header.stamp.sec > 0:
                    delays[topic].append(timestamp * 1e-6 - (msg.header.stamp.sec * 1e3 + msg.header.stamp.nanosec * 1e-6))
                msg_type = topic_types[topic]
                for hdf_name, args in selected_topics[topic].items():
                    if msg_type == "sensor_msgs/msg/CompressedImage":
                        values[topic][hdf_name].append(np.asarray(msg.data).copy())  # decoded below, in one batch
                    elif msg_type == "sensor_msgs/msg/Image":
                        values[topic][hdf_name].append(msg)  # decoded below
                    else:
                        values[topic][hdf_name].append(numeric_values(msg, msg_type, args))

    topic_times, final_data, median_delay = {}, {}, {}
    for topic in selected_topics:
        t = np.asarray(times[topic], dtype=np.float64)
        order = np.argsort(t, kind="stable")  # several bag files, or header stamps: keep time order
        topic_times[topic] = t[order]
        median_delay[topic] = float(np.median(delays[topic])) if delays[topic] else float("nan")
        final_data[topic] = {}
        for hdf_name, items in values[topic].items():
            items = [items[i] for i in order]
            if topic_types[topic] in IMAGE_TYPES:  # messages.image_array: as live consumers decode them
                final_data[topic][hdf_name] = np.array([image_array(item, topic_types[topic], image_size)
                                                        for item in items])
            else:
                arr = np.array(items, dtype=np.float32)
                final_data[topic][hdf_name] = arr[:, None] if arr.ndim == 1 else arr
    return topic_times, final_data, median_delay


def last_index_at(data_times, query_times):
    """Per query time, the index of the last message at or before it (-1: none yet)."""
    return np.searchsorted(data_times, query_times, side="right") - 1


### CREATION UTILS ###
def add_config(fps_used, infos, dir, default=False):

    if default:
        # Default selection of topics in three categories : actions, states, cameras
        action_name, state_name = "motor", "motor"
        action_names, state_names, cameras_names = [], [], []
        action_dims, state_dims, cam_widths, cam_heights = [], [], [], []
        for v in infos:
            hdf5_name, ft_ex = v["hdf5_name"], v["ft_example"]

            if "action" in hdf5_name:
                action_names.append(hdf5_name)
                action_dims.append(ft_ex.shape[0])
                if "joint" in hdf5_name:
                    action_name = "joint"
            elif "cam" in hdf5_name:
                cameras_names.append(hdf5_name)

                if len(ft_ex.shape) < 2:
                    ft_ex = cv2.imdecode(ft_ex, cv2.IMREAD_COLOR)
                height, width = ft_ex.shape[:2]
                cam_widths.append(width)
                cam_heights.append(height)
            else:
                state_names.append(hdf5_name)
                state_dims.append(ft_ex.shape[0])
                if "joint" in hdf5_name:
                    state_name = "joint"

        with open(dir, "w") as config_file:
            action_names_fts = []
            for i, act_name in enumerate(action_names):
                action_names_fts += [act_name.split("/")[-1]+"_"+str(j) for j in range(action_dims[i])]

            state_names_fts = []
            for i, state_name in enumerate(state_names):
                state_names_fts += [state_name.split("/")[-1]+"_"+str(j) for j in range(state_dims[i])]

            new_config_dico = {
                "fps_used":fps_used,
                "outlier_deletion": False,
                "action":{
                    "lerobot_name": action_name,
                    "lerobot_names": action_names_fts,
                    "hdf5_selected_names":action_names
                    },
                "state":{
                    "lerobot_name": state_name,
                    "lerobot_names": state_names_fts,
                    "hdf5_selected_names":state_names
                    },
                "cameras":{cam_name.split("/")[-1]: {"name":cam_name, "width":cam_width, "height":cam_height, "channels":3} for cam_width, cam_height, cam_name in zip(cam_widths, cam_heights, cameras_names)}
            }
            yaml.dump(new_config_dico, config_file, default_flow_style=False)
    else:
        with open(dir) as f:
            list_doc = yaml.safe_load(f)

        for task in fps_used:
            list_doc["fps_used"][task] = fps_used[task]

        with open(dir, "w") as f:
            yaml.dump(list_doc, f, default_flow_style=False)


def read_bag_folder(folder):
    """The bag files (in split order) and the topic types / message counts of one demo folder."""
    bagpaths, topic_types, counts = [], {}, {}
    for file in folder.iterdir():
        if file.suffix in (".db3", ".mcap"):
            bagpaths.append(file.resolve())
        elif file.name in ("metadata.yaml", "metadata.yml"):
            with open(file, "r") as metadata_file:
                metadata = yaml.safe_load(metadata_file)
            for topic_info in metadata["rosbag2_bagfile_information"]["topics_with_message_count"]:
                name = topic_info["topic_metadata"]["name"]
                topic_types[name] = topic_info["topic_metadata"]["type"]
                counts[name] = topic_info["message_count"]
    bagpaths = sorted(bagpaths, key=lambda p: int(p.stem.split('_')[-1]) if p.stem.split('_')[-1].isdigit() else 0)
    return bagpaths, topic_types, counts


def build_demo(bagpaths, topic_types, selected_topics, reference_topic_name, action_time="next_frame",
               time_source="receive", image_size=(256, 256), verbose=False):
    """
    One demo's frames: returns (timestamps (T, 1) s, obs {name: (T, ...)}, actions (T, A), action_part_names,
    raw {name: (times (n,) s, values (n, ...))}, info dict). See the module docstring for the rules.
    """
    if action_time not in ("next_frame", "current"):
        raise ValueError(f"action_time must be next_frame or current, got {action_time}")
    topic_times, topic_data, median_delay = extract_all_topics_single_pass(
        bagpaths, selected_topics, topic_types, time_source=time_source, image_size=image_size, verbose=verbose)

    ref = topic_times[reference_topic_name]
    ref = ref[np.concatenate([[True], np.diff(ref) > 0])]  # one frame per distinct time
    start = max(t[0] for t in topic_times.values())        # every topic has published by then
    frames = ref[ref >= start]
    if action_time == "next_frame":
        obs_times, act_times = frames[:-1], frames[1:]
    else:
        obs_times, act_times = frames, frames
    if len(obs_times) < 2:
        raise ValueError(f"fewer than 2 frames once every topic has published (from {start - ref[0]:.0f} ms)")

    obs, action_parts, action_part_names, raw = {}, [], [], {}
    t0 = obs_times[0]
    for topic, fields in topic_data.items():
        times = topic_times[topic]
        for full_name, arr in fields.items():
            prefix, leaf = full_name.split("/", 1) if "/" in full_name else ("", full_name)
            at = act_times if prefix == "actions" else obs_times
            idx = last_index_at(times, at)
            assert (idx >= 0).all()  # guaranteed by start
            sampled = arr[idx]
            if prefix == "actions":
                action_parts.append(sampled.astype(np.float32, copy=False))
                action_part_names.append(leaf)
            else:
                obs[leaf] = sampled
            if topic_types[topic] in NUMERIC_TYPES:
                raw[full_name] = ((times - t0) * 1e-3, arr)
    actions = (np.concatenate(action_parts, axis=1) if action_parts
               else np.zeros((len(obs_times), 0), dtype=np.float32))
    timestamps = ((obs_times - t0) * 1e-3).astype(np.float32)[:, None]
    dt = np.diff(timestamps[:, 0])
    info = dict(fps_mean=float(np.mean(1.0 / dt)), fps_std=float(np.std(1.0 / dt)),
                dropped_start_s=float((start - ref[0]) * 1e-3) if start > ref[0] else 0.0,
                median_delay_ms={t: d for t, d in median_delay.items()},
                examples={full_name: arr[0] for fields in topic_data.values() for full_name, arr in fields.items()})
    return timestamps, obs, actions, action_part_names, raw, info


def converted_bags(data_root):
    """{bag_name: demo label} of the demos already in the file; removes half-written ones."""
    done = {}
    for label in list(data_root.keys()):
        if label.startswith(TMP_PREFIX):
            print(f"{bcolors.WARNING}Removing the half-written {label} (an interrupted conversion){bcolors.ENDC}")
            del data_root[label]
        elif "bag_name" in data_root[label].attrs:
            done[str(data_root[label].attrs["bag_name"])] = label
    return done


def create_task(dataset_path, desired_path, task, reference_topic_name, selected_topics, verbose=False,
                action_time="next_frame", time_source="receive", image_size=(256, 256), save_raw_streams=True):
    print(f"Processing task {task}")
    bag_folders = sorted(p for p in (dataset_path / task).iterdir() if p.is_dir())
    infos, fps_means, skipped = [], [], []

    with h5py.File(f"{desired_path / task}.h5", "a") as h5file:
        data_root = h5file.require_group("data")
        done = converted_bags(data_root)
        legacy = [k for k in data_root.keys() if "bag_name" not in data_root[k].attrs]
        if legacy:
            raise RuntimeError(f"{desired_path / task}.h5 has demos from the previous converter (no bag_name: "
                               f"{legacy[:3]}...): their order is unknown, so new bags cannot be added to it. "
                               f"Convert into a new file.")
        next_index = len(done)
        for n, folder in enumerate(bag_folders):
            print(f"Processing bag {n + 1}/{len(bag_folders)}: {folder.name}")
            if folder.name in done:
                print(f"     already converted as {done[folder.name]}")
                continue
            start_time = time.time()
            bagpaths, topic_types, counts = read_bag_folder(folder)
            # optional topics (optional: true in the config) the bag lacks are left out of this demo only
            absent = [t for t in selected_topics if t not in topic_types or counts.get(t, 1) == 0]
            optional = [t for t in absent if all(a.get("optional") for a in selected_topics[t].values())]
            if optional:
                print(f"{bcolors.WARNING}     optional topics absent, left out: {optional}{bcolors.ENDC}")
            topics = {t: {h: {k: v for k, v in a.items() if k != "optional"} for h, a in m.items()}
                      for t, m in selected_topics.items() if t not in optional}
            missing = [t for t in topics if t not in topic_types]
            empty = [t for t in topics if t in counts and counts[t] == 0]
            if not bagpaths or missing or empty:
                why = ("no bag file" if not bagpaths else
                       f"topics missing: {missing}" if missing else f"topics that never published: {empty}")
                print(f"{bcolors.FAIL}Error: skipping {folder.name}: {why}{bcolors.ENDC}")
                skipped.append((folder.name, why))
                continue
            if reference_topic_name not in topics:
                raise ValueError(f"the reference topic {reference_topic_name} must be one of the selected topics")
            try:
                timestamps, obs, actions, action_part_names, raw, info = build_demo(
                    bagpaths, topic_types, topics, reference_topic_name, action_time=action_time,
                    time_source=time_source, image_size=image_size, verbose=verbose)
            except ValueError as e:
                print(f"{bcolors.FAIL}Error: skipping {folder.name}: {e}{bcolors.ENDC}")
                skipped.append((folder.name, str(e)))
                continue
            print(f"     fps estimated: {info['fps_mean']:.2f} +/- {info['fps_std']:.2f}, "
                  f"{timestamps.shape[0]} frames, first {info['dropped_start_s']:.2f} s dropped (topics starting)")
            fps_means.append(info["fps_mean"])
            if not infos:
                infos = [{"hdf5_name": k, "ft_example": v} for k, v in info["examples"].items()]

            label = f"demo_{next_index}"
            tmp = data_root.create_group(TMP_PREFIX + label, track_order=True)
            tmp.create_dataset("timestamps", data=timestamps)
            tmp.create_dataset("actions", data=actions)
            obs_grp = tmp.create_group("obs")
            for k, v in obs.items():
                v = np.asarray(v)
                obs_grp.create_dataset(k, data=v.astype(np.uint8, copy=False) if v.dtype == np.uint8
                                       else v.astype(np.float32, copy=False))
            if save_raw_streams:
                raw_grp = tmp.create_group("raw")
                for k, (t, v) in raw.items():
                    raw_grp.create_dataset(k, data=v.astype(np.float32, copy=False))
                    raw_grp.create_dataset(k + "_t", data=t.astype(np.float64))
            tmp.attrs["bag_name"] = folder.name
            tmp.attrs["num_samples"] = int(timestamps.shape[0])
            tmp.attrs["action_time"] = action_time
            tmp.attrs["time_source"] = time_source
            tmp.attrs["dropped_start_s"] = info["dropped_start_s"]
            tmp.attrs["median_header_delay_ms"] = yaml.safe_dump(info["median_delay_ms"])
            if action_part_names:
                tmp.attrs["action_parts"] = ",".join(action_part_names)
            h5file.move(tmp.name, f"data/{label}")  # complete: give it its name
            next_index += 1
            print(f"     data saved as demo '{label}' in '{desired_path / task}.h5' "
                  f"({time.time() - start_time:.2f} s)")

    if skipped:
        print(f"{bcolors.FAIL}{len(skipped)} bag(s) of {task} skipped:{bcolors.ENDC}")
        for name, why in skipped:
            print(f"   {name}: {why}")
    return infos, (float(np.mean(fps_means)) if fps_means else 0.0)
