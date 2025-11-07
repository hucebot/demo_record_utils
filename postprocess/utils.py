"""
Utility functions to handle rosbags.
"""

import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore, get_types_from_msg

from std_msgs.msg import Header

import h5py
import time
from os import walk
import yaml
import re

# Your custom message definition
# check: https://ternaris.gitlab.io/rosbags/examples/register_types.html#from-multiple-files
GRIPPER_WIDTH_MSG = """
std_msgs/Header header
float32 width
"""

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

### EXTRACTION UTILS ### 

def pad_and_concatenate(arrays, axis=0):
    """
    Concatenate along an axis,
    padding other axis with zeros to get equal sizes.

    Args:
        arrays (list[np.ndarray])
        axis (int)

    Returns:
        result (np.ndarray)
    """
    if not arrays:
        raise ValueError("The list is empty.")

    # Verify the number of axis
    ndim = arrays[0].ndim
    if any(a.ndim != ndim for a in arrays):
        raise ValueError("All arrays must have the same number of axis.")

    # Maximal length for each axis
    max_shape = np.max([a.shape for a in arrays], axis=0)

    padded_arrays = []
    for a in arrays:
        # Compute padding for each axis
        pad_width = [(0, max_shape[i] - a.shape[i]) if i != axis else (0, 0) for i in range(ndim)]
        a_padded = np.pad(a, pad_width, mode='constant', constant_values=0)
        padded_arrays.append(a_padded)

    # Final concatenation
    return np.concatenate(padded_arrays, axis=axis)

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


def extractImage(bagpath, topic_name, conversion_args, verbose=False):
    """Extract images from topic of type sensor_msgs/Image as numpy array"""
    if verbose:
        print(f"Extracting '{topic_name}' from '{bagpath}'")

    # Create a type store to use if the bag has no message definitions.
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    # Create a CvBridge to convert between OpenCV Images and ROS Image messages.
    bridge = CvBridge()

    # Create reader instance and open for reading.
    with AnyReader([bagpath], default_typestore=typestore) as reader:
        connections = [x for x in reader.connections if x.topic == topic_name]
        times = []
        images = []
        for connection, timestamp, rawdata in reader.messages(connections=connections):
            msg = reader.deserialize(rawdata, connection.msgtype)
            if msg.encoding == "16UC1":
                msg.encoding = "mono16"
            img_array = bridge.imgmsg_to_cv2(msg)  # [height, width, (channels)]

            times.append(int(timestamp * 1e-6))  # milliseconds
            images.append(img_array)

        image_times = np.array(times)
        images = np.array(images)
        # add a dummy dimension
        image_times = np.expand_dims(image_times, axis=-1)
        if images.ndim == 3:  # depth images
            images = np.expand_dims(images, axis=-1)

        if verbose:
            print("image_times", image_times.shape)
            print("images", images.shape)

        return image_times, [images]


def extractAndEncodeImage(bagpath, topic_name, verbose=False):
    """Extract images from topic of type sensor_msgs/Image as encoded images (JPEG or PNG)"""
    if verbose:
        print(f"Extracting '{topic_name}' from '{bagpath}'")

    # Create a type store to use if the bag has no message definitions.
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    # Create a CvBridge to convert between OpenCV Images and ROS Image messages.
    bridge = CvBridge()

    # Create reader instance and open for reading.
    with AnyReader([bagpath], default_typestore=typestore) as reader:
        connections = [x for x in reader.connections if x.topic == topic_name]
        times = []
        images = []
        max_length = 0
        for connection, timestamp, rawdata in reader.messages(connections=connections):
            times.append(int(timestamp * 1e-6))  # milliseconds
            msg = reader.deserialize(rawdata, connection.msgtype)
            if msg.encoding == "16UC1":
                msg.encoding = "mono16"
            img_array = bridge.imgmsg_to_cv2(msg)  # [height, width, (channels)]

            if img_array.ndim == 3:  # color -> JPEG
                success, encoded_image = cv2.imencode(".jpg", img_array)
            elif img_array.ndim == 2:  # depth -> PNG
                success, encoded_image = cv2.imencode(".png", img_array)
            if not success:
                raise Exception("Image encoding failed!")

            images.append(encoded_image)
            if len(encoded_image) > max_length:
                max_length = len(encoded_image)

        # pad encoded images with 0 to have uniform length
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


def extractCompressedImage(bagpath, topic_name, conversion_args, verbose=False):
    """Extract compressed images from topic of type sensor_msgs/CompressedImage as compressed JPEG"""
    if verbose:
        print(f"Extracting '{topic_name}' from '{bagpath}'")

    # Create a type store to use if the bag has no message definitions.
    typestore = get_typestore(Stores.ROS2_HUMBLE)

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

        # pad encoded image with 0 to have uniform length
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

        return image_times, [padded_images]


def extractAndDecodeCompressedImage(bagpath, topic_name, conversion_args, verbose=False):
    """Extract images from topic of type sensor_msgs/CompressedImage as numpy array"""
    if verbose:
        print(f"Extracting '{topic_name}' from '{bagpath}'")

    # Create a type store to use if the bag has no message definitions.
    typestore = get_typestore(Stores.ROS2_HUMBLE)

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
        if images.ndim == 3:  # depth images
            images = np.expand_dims(images, axis=-1)

        if verbose:
            print("image_times", image_times.shape)
            print("images", images.shape)

        return image_times, [images]


def extractPoseStamped(bagpath, topic_name, conversion_args, verbose=False):
    """Extract 3D poses from topic of type geometry_msgs/PoseStamped as numpy array"""
    if verbose:
        print(f"Extracting '{topic_name}' from '{bagpath}'")

    # Create a type store to use if the bag has no message definitions.
    typestore = get_typestore(Stores.ROS2_HUMBLE)

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

        return pose_times, [pose_array]


def extractTwist(bagpath, topic_name, conversion_args, verbose=False):
    """Extract twist from topic of type geometry_msgs/Twist as numpy array"""
    if verbose:
        print(f"Extracting '{topic_name}' from '{bagpath}'")

    # Create a type store to use if the bag has no message definitions.
    typestore = get_typestore(Stores.ROS2_HUMBLE)

    # Create reader instance and open for reading.
    with AnyReader([bagpath], default_typestore=typestore) as reader:
        connections = [x for x in reader.connections if x.topic == topic_name]

        times = []
        twists = []
        for connection, timestamp, rawdata in reader.messages(connections=connections):
            msg = reader.deserialize(rawdata, connection.msgtype)

            times.append(int(timestamp * 1e-6))
            twists.append(
                np.array(
                    [
                        msg.linear.x,
                        msg.linear.y,
                        msg.linear.z,
                        msg.angular.x,
                        msg.angular.y,
                        msg.angular.z,
                    ]
                )
            )

        twist_times = np.array(times)
        twist_array = np.array(twists, dtype="float32")
        # add a dummy dimension
        twist_times = np.expand_dims(twist_times, axis=-1)

        if verbose:
            print("twist_times", twist_times.shape)
            print("twist_array", twist_array.shape)

        return twist_times, twist_array


def extractGripperFromPointStamped(bagpath, topic_name, conversion_args, verbose=False):
    """Extract gripper command from topic of type geometry_msgs/PointStamped as numpy array"""
    if verbose:
        print(f"Extracting '{topic_name}' from '{bagpath}'")

    # Create a type store to use if the bag has no message definitions.
    typestore = get_typestore(Stores.ROS2_HUMBLE)

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
    
def extractGripperWidth(bagpath, topic_name, conversion_args, verbose=False):
    """Extract gripper width from topic of type custom_msgs/msg/GripperWidth as numpy array"""
    if verbose:
        print(f"Extracting gripper width '{topic_name}' from '{bagpath}'")

    # Create a type store to use if the bag has no message definitions.
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    typestore.register(get_types_from_msg(GRIPPER_WIDTH_MSG, 'custom_msgs/msg/GripperWidth'))


    # Create reader instance and open for reading.
    with AnyReader([bagpath], default_typestore=typestore) as reader:
        connections = [x for x in reader.connections if x.topic == topic_name]

        if not connections:
            if verbose:
                print(f"Topic '{topic_name}' not found. Returning zeros.")
            return None, [None]

        times = []
        data = []
        for connection, timestamp, rawdata in reader.messages(connections=connections):
            msg = reader.deserialize(rawdata, connection.msgtype)

            times.append(int(timestamp * 1e-6))
            data.append(msg.width)

        gripper_times = np.array(times)
        gripper_array = np.array(data, dtype="float32")
        # add a dummy dimension
        gripper_array = np.expand_dims(gripper_array, axis=-1)
        gripper_times = np.expand_dims(gripper_times, axis=-1)

        if verbose:
            print("gripper_times", gripper_times.shape)
            print("gripper_array", gripper_array.shape)

        return gripper_times, [gripper_array]


def extractJointState(bagpath, topic_name, conversion_args, verbose=False):
    """Extract color images from topic of type sensor_msgs/JointState as numpy array"""
    if verbose:
        print(f"Extracting '{topic_name}' from '{bagpath}'")

    # Create a type store to use if the bag has no message definitions.
    typestore = get_typestore(Stores.ROS2_HUMBLE)

    # Create reader instance and open for reading.
    with AnyReader([bagpath], default_typestore=typestore) as reader:
        connections = [x for x in reader.connections if x.topic == topic_name]
        times = []
        data = [[] for _ in conversion_args]
        
        for connection, timestamp, rawdata in reader.messages(connections=connections):
            msg = reader.deserialize(rawdata, connection.msgtype)
            
            #print()

            times.append(int(timestamp * 1e-6))  # milliseconds

            for i, arg in enumerate(conversion_args):
                if not arg:
                    data[i].append(msg.position)
                elif arg["physical_quantity"] == "position":
                    data[i].append(msg.position)
                elif arg["physical_quantity"] == "velocity":
                    data[i].append(msg.velocity)
                elif arg["physical_quantity"] == "effort":
                    data[i].append(msg.effort)
                else:
                    raise NotImplementedError


        joint_times = np.array(times)
        joint_data = [np.array(data[i], dtype="float32") for i in range(len(data))]

        # add a dummy dimension
        joint_times = np.expand_dims(joint_times, axis=-1)

        if verbose:
            print("joint_times", joint_times.shape)
            print("joint_data", [joint_data[i].shape for i in range(len(joint_data))])

        return joint_times, joint_data
    
def extractWrenchStamped(bagpath, topic_name, conversion_args, verbose=False):
    """Extract color images from topic of type sensor_msgs/JointState as numpy array"""
    if verbose:
        print(f"Extracting '{topic_name}' from '{bagpath}'")

    # Create a type store to use if the bag has no message definitions.
    typestore = get_typestore(Stores.ROS2_HUMBLE)

    # Create reader instance and open for reading.
    with AnyReader([bagpath], default_typestore=typestore) as reader:
        connections = [x for x in reader.connections if x.topic == topic_name]
        times = []
        data = [[] for _ in conversion_args]
        
        for connection, timestamp, rawdata in reader.messages(connections=connections):
            msg = reader.deserialize(rawdata, connection.msgtype)
            
            #print()

            times.append(int(timestamp * 1e-6))  # milliseconds
            
            for i, arg in enumerate(conversion_args):
                if not arg:
                    data[i].append([msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z])
                elif arg["physical_quantity"] == "force":
                    data[i].append([msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z])
                elif arg["physical_quantity"] == "torque":
                    data[i].append([msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z])
                else:
                    raise NotImplementedError

        joint_times = np.array(times)
        joint_data = [np.array(data[i], dtype="float32") for i in range(len(data))]

        # add a dummy dimension
        joint_times = np.expand_dims(joint_times, axis=-1)

        if verbose:
            print("joint_times", joint_times.shape)
            print("joint_data", [joint_data[i].shape for i in range(len(joint_data))])

        return joint_times, joint_data

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


def getLastDataAtRefTimes(reference_times, data_times, data_dict):
    indices = []
    idx = 0
    for ref_time in reference_times:
        while idx < len(data_times) - 1 and data_times[idx + 1] <= ref_time:
            idx += 1
        indices.append(idx)

    last_data_dict = {}
    for key in data_dict.keys():
        last_data_dict[key] = data_dict[key][indices]

    return last_data_dict

def extract_topic(topic_type, bagpaths, topic_name, topic_conversions, verbose=True):
    conversion_args = [topic_conversions[hdf5_name] for hdf5_name in topic_conversions.keys()]
    timestamps, results = [], [[] for _ in conversion_args]
    for bagpath in bagpaths:
        timestamp, result = None, None
        if topic_type == "sensor_msgs/msg/CompressedImage":
            timestamp, result = extractCompressedImage(bagpath, topic_name, conversion_args, verbose=verbose)
        elif topic_type == "sensor_msgs/msg/JointState":
            timestamp, result = extractJointState(bagpath, topic_name, conversion_args, verbose=verbose)
        elif topic_type == "geometry_msgs/msg/PoseStamped":
            timestamp, result = extractPoseStamped(bagpath, topic_name, conversion_args, verbose=verbose)
        elif topic_type == "geometry_msgs/msg/WrenchStamped":
            timestamp, result = extractWrenchStamped(bagpath, topic_name, conversion_args, verbose=verbose)
        elif topic_type == "custom_msgs/msg/GripperWidth":
            timestamp, result = extractGripperWidth(bagpath, topic_name, conversion_args, verbose=verbose)
        elif topic_type == "sensor_msgs/msg/Image":
            timestamp, result = extractImage(bagpath, topic_name, conversion_args, verbose=verbose)
        else:
            raise NotImplementedError(f"The topic type {topic_type} is not recognized")
        
        if timestamp is None:
            timestamps.append(timestamps[-1][None,-1])
        else:
            timestamps.append(timestamp)
        for i in range(len(conversion_args)):
            if result[i] is None:
                results[i].append(results[i][-1][None,-1])
            else:
                results[i].append(result[i])
    
    result = [pad_and_concatenate(results[idx], axis=0) for idx in range(len(results))]
    timestamp = np.concatenate(timestamps, axis=0)

    if verbose:
        print("final extracted result", {hdf5_name:result[idx].shape for idx, hdf5_name in enumerate(topic_conversions.keys())})
    
    return timestamp, {hdf5_name:result[idx] for idx, hdf5_name in enumerate(topic_conversions.keys())}

### CREATION UTILS ### 
def create_default_config(fps_used, infos, dir):
    
    # Default selection of topics in three categories : actions, states, cameras
    action_name, state_name = "motor", "motor"
    action_names, state_names, cameras_names = [], [], []
    action_dims, state_dims, cameras_new_names = [], [], []
    for v in infos:
        hdf5_name, ft_dim = v["hdf5_name"], v["ft_dim"]

        if "action" in hdf5_name:
            action_names.append(hdf5_name)
            action_dims.append(ft_dim)
            if "joint" in hdf5_name:
                action_name = "joint"
        elif "cam" in hdf5_name:
            cameras_names.append(hdf5_name)
            cameras_new_names.append(hdf5_name.split("/")[-1])
        else:
            state_names.append(hdf5_name)
            state_dims.append(ft_dim)
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
            "cameras":{new_cam_name: {"name":cam_name} for new_cam_name, cam_name in zip(cameras_new_names, cameras_names)}
        }
        yaml.dump(new_config_dico, config_file, default_flow_style=False)

def create_task(dataset_path, desired_path, task, reference_topic_name, selected_topics, verbose=False):

    print(f"Processing task {task}")

    # Read the name of each episode for this task
    ep_names = next(walk(dataset_path / task))[1]
    num_bags = len(ep_names)

    infos = []
    number_topics = len(selected_topics.keys())
    fps_tot_mean = 0

    for demo_idx, demo_folder in enumerate(ep_names):
        print(f"Processing demo {demo_idx + 1}/{num_bags}")
        demo_label = f"{demo_idx:03d}"
        demo_start_time = time.time()

        # Read metadata
        bagpaths, topic_types = [], {}
        for file in (dataset_path / task / demo_folder).iterdir():
            if str(file)[-4:] == ".db3":
                bagpaths.append(file.resolve())

            elif str(file)[-5:] == ".yaml" or str(file)[-4:] == ".yml":
                # Read metadata file to get each topic and its type
                with open(file, "r") as metadata_file:
                    metadata = yaml.safe_load(metadata_file)
        
                    for topic_info in metadata["rosbag2_bagfile_information"]["topics_with_message_count"]:
                        topic_types[topic_info["topic_metadata"]["name"]] = topic_info["topic_metadata"]["type"]

                    # Check inconsistencies
                    for topic_name in selected_topics.keys():
                        if not(topic_name in topic_types.keys()):
                            print(f"{bcolors.FAIL}Error : The given topic {topic_name} is unavailable in the data so the conversion cannot proceed !{bcolors.ENDC}")
                            return infos, fps_tot_mean
        
        # Sort to get everything in the right order
        bagpaths = sorted(bagpaths, key=lambda p: int(p.stem.split('_')[-1]))

        # Search for reference topic
        reference_topic_times = None
        for i, topic_name in enumerate(selected_topics.keys()):
            if topic_name == reference_topic_name:
                topic_times, _ = extract_topic(topic_types[topic_name], bagpaths, topic_name, selected_topics[topic_name], verbose=verbose)
                reference_topic_times = topic_times
                break
            elif i == number_topics-1:
                if demo_idx == 0:
                    print(f"{bcolors.WARNING}Warning : The given reference topic is unavailable in the data so the last topic is considered the reference topic by default !{bcolors.ENDC}")

                topic_times, _ = extract_topic(topic_types[topic_name], bagpaths, topic_name, selected_topics[topic_name], verbose=verbose)
                reference_topic_times = topic_times

        timestamps = reference_topic_times - reference_topic_times[0]
        timestamps = timestamps * 1e-3
        timestamps = timestamps.astype("float32")

        # Compute an estimated fps
        timestamps = timestamps[np.pad(np.diff(timestamps[:,0]), (0,1), mode='constant', constant_values=1) > 0]
        fps_mean, fps_std = np.mean(1/np.diff(timestamps[:,0])).item(), np.std(1/np.diff(timestamps[:,0])).item()
        print("fps estimated : ", fps_mean, "+/-", fps_std)
        fps_tot_mean = (fps_tot_mean*demo_idx+fps_mean)/(demo_idx+1)



        with h5py.File(f"{desired_path / task}.h5", "a") as h5file:
            group = h5file.create_group(demo_label, track_order=True)
            group.create_dataset("timestamps", data=timestamps)

            for topic_name in selected_topics.keys():
                # Open rosbag and extract topic data.
                topic_times, topic_data = extract_topic(topic_types[topic_name], bagpaths, topic_name, selected_topics[topic_name], verbose=verbose)

                # Synch data with given topic timestamps.
                synch_topic_data = getLastDataAtRefTimes(reference_topic_times, topic_times, topic_data)
                
                for hdf5_name in topic_data.keys():
                    # Get the infos if it is the first demonstration
                    if demo_idx == 0:
                        infos.append({})
                        infos[-1]["hdf5_name"] = hdf5_name
                        infos[-1]["ft_dim"] = topic_data[hdf5_name].shape[1]

                    group.create_dataset(hdf5_name, data=synch_topic_data[hdf5_name])

        print(f"     data saved as demo '{demo_label}' in '{desired_path / task}.h5' file")
        print(f"     time: {(time.time() - demo_start_time):.2f} seconds")

    # Returns important metadata for default config
    return infos, fps_tot_mean

### UNIT TESTS ###
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