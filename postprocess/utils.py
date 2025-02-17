"""
Utility functions to handle rosbags.
"""

import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore


def fixed_compressed_imgmsg_to_cv2(cmprs_img_msg, desired_encoding="passthrough"):
    """
    Convert a sensor_msgs::CompressedImage message to an OpenCV :cpp:type:`cv::Mat`.

    :param cmprs_img_msg:   A :cpp:type:`sensor_msgs::CompressedImage` message
    :param desired_encoding:  The encoding of the image data, one of the following strings:

        * ``"passthrough"``
        * one of the standard strings in sensor_msgs/image_encodings.h

    :rtype: :cpp:type:`cv::Mat`
    :raises CvBridgeError: when conversion is not possible.

    If desired_encoding is ``"passthrough"``, then the returned image has the same format
    as img_msg. Otherwise desired_encoding must be one of the standard image encodings

    This function returns an OpenCV :cpp:type:`cv::Mat` message on success,
    or raises :exc:`cv_bridge.CvBridgeError` on failure.

    If the image only has one channel, the shape has size 2 (width and height)
    """
    import cv2
    import numpy as np

    str_msg = cmprs_img_msg.data
    buf = np.ndarray(shape=(1, len(str_msg)), dtype=np.uint8, buffer=cmprs_img_msg.data)
    im = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)

    if desired_encoding == "passthrough":
        return im

    from cv_bridge.boost.cv_bridge_boost import cvtColor2

    try:
        res = cvtColor2(im, "bgr8", desired_encoding)
    except RuntimeError as e:
        raise CvBridgeError(e)

    return res


def extractImage(bagpath, topic_name, verbose=False):
    """Extract images from topic of type sensor_msgs/Image as numpy array"""
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
            img_array = bridge.imgmsg_to_cv2(msg)  # [height, width, (channels)]

            times.append(int(timestamp * 1e-6))  # milliseconds
            images.append(img_array)

        image_times = np.array(times)
        images = np.array(images)
        # add a dummy dimension
        image_times = np.expand_dims(image_times, axis=-1)
        if images.ndim == 2:  # depth images
            images = np.expand_dims(images, axis=-1)

        if verbose:
            print("image_times", image_times.shape)
            print("images", images.shape)

        return image_times, images


def extractCompressedImage(bagpath, topic_name, verbose=False):
    """Extract compressed images from topic of type sensor_msgs/CompressedImage as compressed JPEG"""
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


def extractAndDecodeCompressedImage(bagpath, topic_name, verbose=False):
    """Extract images from topic of type sensor_msgs/CompressedImage as numpy array"""
    if verbose:
        print(f"Extracting '{topic_name}' from '{bagpath}'")

    # Create a type store to use if the bag has no message definitions.
    typestore = get_typestore(Stores.ROS1_NOETIC)

    # Create reader instance and open for reading.
    with AnyReader([bagpath], default_typestore=typestore) as reader:
        connections = [x for x in reader.connections if x.topic == topic_name]
        times = []
        images = []
        for connection, timestamp, rawdata in reader.messages(connections=connections):
            msg = reader.deserialize(rawdata, connection.msgtype)
            img_array = fixed_compressed_imgmsg_to_cv2(msg)  # [height, width, channels]

            times.append(int(timestamp * 1e-6))  # milliseconds
            images.append(img_array)

        image_times = np.array(times)
        images = np.array(images)
        # add a dummy dimension
        image_times = np.expand_dims(image_times, axis=-1)
        if images.ndim == 2:  # depth images
            images = np.expand_dims(images, axis=-1)

        if verbose:
            print("image_times", image_times.shape)
            print("images", images.shape)

        return image_times, images


def extractPoseStamped(bagpath, topic_name, verbose=False):
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


def extractGripperFromPointStamped(bagpath, topic_name, verbose=False):
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


def extractJointState(bagpath, topic_name, verbose=False):
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


def getLastDataAtRefTimes(reference_times, data_times, data_array):
    indices = []
    idx = 0
    for ref_time in reference_times:
        while idx < len(data_times) - 1 and data_times[idx + 1] <= ref_time:
            idx += 1
        indices.append(idx)

    last_data_array = data_array[indices]

    return last_data_array


if __name__ == "__main__":
    """Test fixed_compressed_imgmsg_to_cv2"""
    # create a 16bit depth image
    im0 = np.empty(shape=(100, 100), dtype=np.uint16)
    im0[:] = 2500  # 2.5m
    print("original:", np.max(im0), im0.dtype)
    # convert to compressed message
    msg = CvBridge().cv2_to_compressed_imgmsg(im0, dst_format="png")
    # convert back to numpy array
    im1 = fixed_compressed_imgmsg_to_cv2(msg)
    print("fixed converted:", np.max(im1), im1.dtype)
    print("match?", np.all(im0 == im1))
    im2 = CvBridge().compressed_imgmsg_to_cv2(msg)
    print("standard converted:", np.max(im2), im2.dtype)
    print("match?", np.all(im0 == im2))
