"""
Tests of the rosbag -> HDF5 conversion (utils.py) on synthetic bags whose values are their own times, so every
alignment error shows as a number:  pip install rosbags h5py opencv-python-headless pytest; pytest postprocess/

Bag layout (times in s from the bag's start): camera frames at 30 Hz from 0.10 s (the reference), the desired pose at
100 Hz from 0 s with x = its time, the wrist force at 800 Hz from 0 s with fx = its time, a gripper command published
once at 0.30 s (an event topic: frames before it have no command and are dropped).
"""
import pathlib
import sys

import cv2
import h5py
import numpy as np
import pytest

rosbags = pytest.importorskip("rosbags")
from rosbags.rosbag2 import Writer  # noqa: E402
from rosbags.typesys import Stores, get_typestore  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from utils import create_task  # noqa: E402

TS = get_typestore(Stores.ROS2_HUMBLE)
T0 = 1_700_000_000 * 10**9  # ns, the bag's start
CAM, POSE, FORCE, GRIP = ("/webcam1/image_raw/compressed", "/cartesian_impedance/equilibrium_pose",
                          "/bota_ft_sensor/wrench_filtered", "/panda_gripper/gripper_command")
SELECTED = {
    CAM: {"observations/front_cam1": {}},
    FORCE: {"observations/ee_force": {"physical_quantity": "force"}},
    POSE: {"actions/desired_ee_pos": {}},
    GRIP: {"actions/desired_gripper_width": {}},
}


def header(t_ns):
    Header, Time = TS.types["std_msgs/msg/Header"], TS.types["builtin_interfaces/msg/Time"]
    return Header(stamp=Time(sec=int(t_ns // 10**9), nanosec=int(t_ns % 10**9)), frame_id="")


def write_bag(folder, duration=1.0, topics=(CAM, POSE, FORCE, GRIP), law=None):
    msgs = []  # (t_ns, topic, type, msg)
    if law is not None:  # the force law's settings, published once a second (ForceVAM's law bridge)
        for t in np.arange(0.05, duration, 1.0):
            msgs.append((t, "/forcevam/law", "std_msgs/msg/String", lambda ns: TS.types["std_msgs/msg/String"](data=law)))
        topics = tuple(topics) + ("/forcevam/law",)
    jpeg = np.asarray(cv2.imencode(".jpg", np.full((8, 8, 3), 128, np.uint8))[1]).reshape(-1)
    for t in np.arange(0.10, duration, 1 / 30):
        msgs.append((t, CAM, "sensor_msgs/msg/CompressedImage",
                     lambda ns: TS.types["sensor_msgs/msg/CompressedImage"](header=header(ns), format="jpeg", data=jpeg)))
    P, Q, Pose, PS = (TS.types["geometry_msgs/msg/Point"], TS.types["geometry_msgs/msg/Quaternion"],
                      TS.types["geometry_msgs/msg/Pose"], TS.types["geometry_msgs/msg/PoseStamped"])
    for t in np.arange(0.0, duration, 1 / 100):
        msgs.append((t, POSE, "geometry_msgs/msg/PoseStamped", lambda ns, t=t: PS(
            header=header(ns), pose=Pose(position=P(x=float(t), y=0.0, z=0.0), orientation=Q(x=0.0, y=0.0, z=0.0, w=1.0)))))
    V, W, WS = TS.types["geometry_msgs/msg/Vector3"], TS.types["geometry_msgs/msg/Wrench"], TS.types["geometry_msgs/msg/WrenchStamped"]
    for t in np.arange(0.0, duration, 1 / 800):
        msgs.append((t, FORCE, "geometry_msgs/msg/WrenchStamped", lambda ns, t=t: WS(
            header=header(ns), wrench=W(force=V(x=float(t), y=0.0, z=0.0), torque=V(x=0.0, y=0.0, z=0.0)))))
    PtS = TS.types["geometry_msgs/msg/PointStamped"]
    msgs.append((0.30, GRIP, "geometry_msgs/msg/PointStamped",
                 lambda ns: PtS(header=header(ns), point=P(x=0.08, y=0.0, z=0.0))))
    msgs.sort(key=lambda m: m[0])
    with Writer(folder, version=8) as writer:
        conns = {}
        for t, topic, msgtype, make in msgs:
            if topic not in topics:
                continue
            if topic not in conns:
                conns[topic] = writer.add_connection(topic, msgtype, typestore=TS)
            ns = T0 + int(round(t * 1e9))
            writer.write(conns[topic], ns, TS.serialize_cdr(make(ns), msgtype))


def convert(root, out, **kw):
    return create_task(root, out, "task", CAM, SELECTED, **kw)


@pytest.fixture
def bags(tmp_path):
    root = tmp_path / "bags"
    (root / "task").mkdir(parents=True)
    write_bag(root / "task" / "task_20260101_120000")
    write_bag(root / "task" / "task_20260101_110000", duration=0.8)  # recorded first
    return root


def test_frames_actions_and_raw_streams(bags, tmp_path):
    convert(bags, tmp_path)
    with h5py.File(tmp_path / "task.h5") as f:
        assert [f[f"data/demo_{k}"].attrs["bag_name"] for k in range(2)] == ["task_20260101_110000",
                                                                            "task_20260101_120000"]
        d = f["data/demo_1"]
        t = d["timestamps"][:, 0]
        frames = 0.10 + np.arange(30) / 30
        first = frames[frames >= 0.30][0]           # the first frame once the gripper command exists
        frame_times = first + t                     # back to bag time
        assert frame_times[0] == pytest.approx(first, abs=1e-3)
        assert d.attrs["dropped_start_s"] == pytest.approx(first - 0.10, abs=1e-3)
        # observations: the last force at or before the frame (800 Hz)
        fx = d["obs/ee_force"][:, 0]
        assert np.all(fx <= frame_times + 1e-6) and np.all(frame_times - fx < 1 / 800 + 1e-6)
        # actions: the command in effect at the NEXT frame (the last frame has none and is dropped)
        x = d["actions"][:, 0]
        nxt = frame_times + 1 / 30
        assert np.all(x <= nxt + 1e-6) and np.all(nxt - x < 1 / 100 + 1e-6)
        assert np.all(x > frame_times)              # never the command the observation was already following
        assert d["actions"].shape == (len(t), 8) and np.allclose(d["actions"][:, 7], 0.08)
        assert d.attrs["action_parts"] == "desired_ee_pos,desired_gripper_width"
        # every force message, at full rate, on the frames' clock
        rt, rv = d["raw/observations/ee_force_t"][:], d["raw/observations/ee_force"][:, 0]
        assert len(rt) == 800 and np.all(np.diff(rt) > 0)
        assert np.allclose(rv, rt + first, atol=1e-5)
        assert d["obs/front_cam1"].shape[1:] == (256, 256, 3) and d["obs/front_cam1"].dtype == np.uint8


def test_current_action_time(bags, tmp_path):
    convert(bags, tmp_path, action_time="current")
    with h5py.File(tmp_path / "task.h5") as f:
        d = f["data/demo_1"]
        first = (0.10 + np.arange(30) / 30)[(0.10 + np.arange(30) / 30) >= 0.30][0]
        frame_times = first + d["timestamps"][:, 0]
        x = d["actions"][:, 0]
        assert np.all(x <= frame_times + 1e-6) and np.all(frame_times - x < 1 / 100 + 1e-6)


def test_reconverting_adds_only_new_bags(bags, tmp_path):
    convert(bags, tmp_path)
    write_bag(bags / "task" / "task_20260101_130000")
    with h5py.File(tmp_path / "task.h5", "a") as f:  # an interrupted conversion left a half-written demo
        f.create_group("data/_incomplete_demo_2")
    convert(bags, tmp_path)
    with h5py.File(tmp_path / "task.h5") as f:
        names = sorted(f["data"].keys())
        assert names == ["demo_0", "demo_1", "demo_2"]
        assert f["data/demo_2"].attrs["bag_name"] == "task_20260101_130000"


def test_a_bag_missing_a_topic_is_skipped(bags, tmp_path):
    write_bag(bags / "task" / "task_20260101_115000", topics=(CAM, POSE, FORCE))  # no gripper command
    convert(bags, tmp_path)
    with h5py.File(tmp_path / "task.h5") as f:
        assert [f[f"data/demo_{k}"].attrs["bag_name"] for k in range(len(f["data"]))] == [
            "task_20260101_110000", "task_20260101_120000"]


def test_optional_topic_absent(bags, tmp_path):
    selected = dict(SELECTED, **{"/bota_ft_sensor/wrench": {"observations/ee_force_raw": {
        "physical_quantity": "force", "optional": True}}})
    create_task(bags, tmp_path, "task", CAM, selected)
    with h5py.File(tmp_path / "task.h5") as f:
        assert len(f["data"]) == 2 and "ee_force_raw" not in f["data/demo_0/obs"]


def test_attribute_topics(tmp_path):
    """A text topic's last message becomes a demo attribute (the force law's settings); absent: no attribute."""
    root = tmp_path / "bags"
    (root / "task").mkdir(parents=True)
    law = '{"k_rest": 1500.0, "push": 40.0}'
    write_bag(root / "task" / "task_20260101_100000", law=law)
    write_bag(root / "task" / "task_20260101_110000")
    attrs = [{"hdf5_attr": "env_law", "from_rosbag_topic_name": "/forcevam/law"}]
    convert(root, tmp_path, selected_attributes=attrs)
    with h5py.File(tmp_path / "task.h5") as f:
        assert f["data/demo_0"].attrs["env_law"] == law
        assert "env_law" not in f["data/demo_1"].attrs

