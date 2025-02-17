"""
Utility functions to handle rosbags.
"""

import cv2
from cv_bridge import CvBridge
import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore


def extract_compressed(bagpath, topic_name, verbose=False):
    """Extract compressed image from topic of type sensor_msgs/CompressedImage as numpy array"""
    if verbose:
        print(f"Extracting '{topic_name}' from '{bagpath}'")

    # Create a type store to use if the bag has no message definitions.
    typestore = get_typestore(Stores.ROS1_NOETIC)

    # Create reader instance and open for reading.
    with AnyReader([bagpath], default_typestore=typestore) as reader:
        connections = [x for x in reader.connections if x.topic == topic_name]
        times = []
        images = []
        max_length = 0
        for connection, timestamp, rawdata in reader.messages(connections=connections):
            msg = reader.deserialize(rawdata, connection.msgtype)
            times.append(int(timestamp * 1e-6))  # milliseconds
            images.append(msg.data)
            if len(msg.data) > max_length:
                max_length = len(msg.data)

        # pad jpg with 0 to have uniform length
        padded_images = []
        for img in images:
            padded_images.append(
                np.append(img, np.zeros((max_length - len(img),), dtype=img.dtype))
            )

        image_times = np.array(times)
        padded_images = np.array(padded_images)

        # add a dummy dimension
        image_times = np.expand_dims(image_times, axis=-1)

        if verbose:
            print("image_times", image_times.shape)
            print("padded_images", padded_images.shape)

        return image_times, padded_images


def extract_color_from_compressed(bagpath, topic_name, verbose=False):
    """Extract color images from topic of type sensor_msgs/CompressedImage as numpy array"""
    if verbose:
        print(f"Extracting '{topic_name}' from '{bagpath}'")

    # Create a type store to use if the bag has no message definitions.
    typestore = get_typestore(Stores.ROS1_NOETIC)
    # Create a CvBridge to convert between OpenCV Images and ROS Image messages.
    bridge = CvBridge()

    # Create reader instance and open for reading.
    with AnyReader([bagpath], default_typestore=typestore) as reader:
        connections = [x for x in reader.connections if x.topic == topic_name]
        times = []
        images = []
        for connection, timestamp, rawdata in reader.messages(connections=connections):
            msg = reader.deserialize(rawdata, connection.msgtype)
            img_array = bridge.compressed_imgmsg_to_cv2(
                msg
            )  # [height, width, channels(BGR)]

            times.append(int(timestamp * 1e-6))  # milliseconds
            images.append(img_array)

        color_times = np.array(times)
        color_images = np.array(images)
        # add a dummy dimension
        color_times = np.expand_dims(color_times, axis=-1)

        if verbose:
            print("color_times", color_times.shape)
            print("color_images", color_images.shape)

        return color_times, color_images


def extract_depth_from_compressed(bagpath, topic_name, verbose=False):
    """Extract depth images from topic of type sensor_msgs/CompressedImage as numpy array"""
    if verbose:
        print(f"Extracting '{topic_name}' from '{bagpath}'")

    # Create a type store to use if the bag has no message definitions.
    typestore = get_typestore(Stores.ROS1_NOETIC)
    # Create a CvBridge to convert between OpenCV Images and ROS Image messages.
    bridge = CvBridge()

    # Create reader instance and open for reading.
    with AnyReader([bagpath], default_typestore=typestore) as reader:
        connections = [x for x in reader.connections if x.topic == topic_name]
        times = []
        images = []
        for connection, timestamp, rawdata in reader.messages(connections=connections):
            msg = reader.deserialize(rawdata, connection.msgtype)
            img_array = bridge.compressed_imgmsg_to_cv2(msg)  # [height, width]

            times.append(int(timestamp * 1e-6))  # milliseconds
            images.append(img_array)

        depth_times = np.array(times)
        depth_images = np.array(images)
        # add a dummy dimension
        depth_times = np.expand_dims(depth_times, axis=-1)
        depth_images = np.expand_dims(depth_images, axis=-1)

        if verbose:
            print("depth_times", depth_times.shape)
            print("depth_images", depth_images.shape)

        return depth_times, depth_images


def extract_depth(bagpath, topic_name, verbose=False):
    """Extract depth images from topic of type sensor_msgs/CompressedImage as numpy array"""
    if verbose:
        print(f"Extracting '{topic_name}' from '{bagpath}'")

    # Create a type store to use if the bag has no message definitions.
    typestore = get_typestore(Stores.ROS1_NOETIC)
    # Create a CvBridge to convert between OpenCV Images and ROS Image messages.
    bridge = CvBridge()

    # Create reader instance and open for reading.
    with AnyReader([bagpath], default_typestore=typestore) as reader:
        connections = [x for x in reader.connections if x.topic == topic_name]
        times = []
        images = []
        for connection, timestamp, rawdata in reader.messages(connections=connections):
            msg = reader.deserialize(rawdata, connection.msgtype)
            img_array = bridge.imgmsg_to_cv2(msg)  # [height, width]

            times.append(int(timestamp * 1e-6))  # milliseconds
            images.append(img_array)

        depth_times = np.array(times)
        depth_images = np.array(images)
        # add a dummy dimension
        depth_times = np.expand_dims(depth_times, axis=-1)
        depth_images = np.expand_dims(depth_images, axis=-1)

        if verbose:
            print("depth_times", depth_times.shape)
            print("depth_images", depth_images.shape)

        return depth_times, depth_images


def extract_pose_stamped(bagpath, topic_name, verbose=False):
    """Extract 3D poses from topic of type geometry_msgs/PoseStamped as numpy array"""
    if verbose:
        print(f"Extracting '{topic_name}' from '{bagpath}'")

    # Create a type store to use if the bag has no message definitions.
    typestore = get_typestore(Stores.ROS1_NOETIC)

    # Create reader instance and open for reading.
    with AnyReader([bagpath], default_typestore=typestore) as reader:
        connections = [x for x in reader.connections if x.topic == topic_name]

        times = []
        poses = []
        for connection, timestamp, rawdata in reader.messages(connections=connections):
            msg = reader.deserialize(rawdata, connection.msgtype)

            times.append(int(timestamp * 1e-6))
            poses.append(
                np.array(
                    [
                        msg.pose.position.x,
                        msg.pose.position.y,
                        msg.pose.position.z,
                        msg.pose.orientation.x,
                        msg.pose.orientation.y,
                        msg.pose.orientation.z,
                        msg.pose.orientation.w,
                    ]
                )
            )

        pose_times = np.array(times)
        pose_array = np.array(poses, dtype="float32")
        # add a dummy dimension
        pose_times = np.expand_dims(pose_times, axis=-1)

        if verbose:
            print("pose_times", pose_times.shape)
            print("pose_array", pose_array.shape)

        return pose_times, pose_array


def extract_gripper_from_point_stamped(bagpath, topic_name, verbose=False):
    """Extract gripper command from topic of type geometry_msgs/PointStamped as numpy array"""
    if verbose:
        print(f"Extracting '{topic_name}' from '{bagpath}'")

    # Create a type store to use if the bag has no message definitions.
    typestore = get_typestore(Stores.ROS1_NOETIC)

    # Create reader instance and open for reading.
    with AnyReader([bagpath], default_typestore=typestore) as reader:
        connections = [x for x in reader.connections if x.topic == topic_name]

        times = []
        data = []
        for connection, timestamp, rawdata in reader.messages(connections=connections):
            msg = reader.deserialize(rawdata, connection.msgtype)

            times.append(int(timestamp * 1e-6))
            data.append(msg.point.x)

        gripper_times = np.array(times)
        gripper_array = np.array(data, dtype="float32")
        # add a dummy dimension
        gripper_array = np.expand_dims(gripper_array, axis=-1)
        gripper_times = np.expand_dims(gripper_times, axis=-1)

        if verbose:
            print("gripper_times", gripper_times.shape)
            print("gripper_array", gripper_array.shape)

        return gripper_times, gripper_array


def extract_joint_state(bagpath, topic_name, verbose=False):
    """Extract color images from topic of type sensor_msgs/JointState as numpy array"""
    if verbose:
        print(f"Extracting '{topic_name}' from '{bagpath}'")

    # Create a type store to use if the bag has no message definitions.
    typestore = get_typestore(Stores.ROS1_NOETIC)

    # Create reader instance and open for reading.
    with AnyReader([bagpath], default_typestore=typestore) as reader:
        connections = [x for x in reader.connections if x.topic == topic_name]
        times = []
        positions = []
        velocities = []
        for connection, timestamp, rawdata in reader.messages(connections=connections):
            msg = reader.deserialize(rawdata, connection.msgtype)
            print()

            times.append(int(timestamp * 1e-6))  # milliseconds
            positions.append(msg.position)
            velocities.append(msg.velocity)

        joint_times = np.array(times)
        joint_positions = np.array(positions, dtype="float32")
        joint_velocities = np.array(velocities, dtype="float32")
        # add a dummy dimension
        joint_times = np.expand_dims(joint_times, axis=-1)

        if verbose:
            print("joint_times", joint_times.shape)

        return joint_times, joint_positions, joint_velocities


def save_mp4_from_imgs(output_file, fps, imgs, color=True):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for MP4 files

    # Get video dimensions from the first image
    images = imgs
    num_imgs, height, width, _ = imgs.shape

    # Create VideoWriter
    if color:
        out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
    else:
        out = cv2.VideoWriter(output_file, fourcc, fps, (width, height), 0)
    # Write each frame to the video
    for i in range(num_imgs):
        out.write(images[i, :])

    # Release resources
    out.release()


def get_last_data_at_ref_times(reference_times, data_times, data_array):
    indices = []
    idx = 0
    for ref_time in reference_times:
        while idx < len(data_times) - 1 and data_times[idx + 1] <= ref_time:
            idx += 1
        indices.append(idx)

    last_data_array = data_array[indices]

    return last_data_array
