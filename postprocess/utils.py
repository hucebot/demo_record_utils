"""
Utility functions to handle rosbags.
"""

import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore
from collections import defaultdict, deque
from bisect import bisect_right


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

        return image_times, images


def extractAndEncodeImage(bagpath, topic_name, verbose=False):
    """Extract images from topic of type sensor_msgs/Image as encoded images (JPEG or PNG)"""
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
        if images.ndim == 3:  # depth images
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


def extractTwist(bagpath, topic_name, verbose=False):
    """Extract twist from topic of type geometry_msgs/Twist as numpy array"""
    if verbose:
        print(f"Extracting '{topic_name}' from '{bagpath}'")

    # Create a type store to use if the bag has no message definitions.
    typestore = get_typestore(Stores.ROS1_NOETIC)

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

def extractTF(bagpath, topic_name, parent_frame, child_frame, verbose=False):
    """
    Extract transforms from TF topics (tf2_msgs/TFMessage), both dynamic and static,
    composing intermediate chains to return the pose of child_frame in parent_frame.

    Optimized by grouping transforms per segment and using binary search per timestamp.

    Returns:
      - tf_times: np.ndarray (N,1) of timestamps in ms
      - tf_array: np.ndarray (N,7) of [tx, ty, tz, qx, qy, qz, qw]

    Raises RuntimeError if no complete transform chain is found.
    """

    # Quaternion<->matrix helpers (omitted for brevity, same as before)...
    def quaternion_matrix(q):
        x,y,z,w = q; xx,yy,zz = x*x,y*y,z*z; xy,xz,yz = x*y, x*z, y*z; wx,wy,wz = w*x, w*y, w*z
        M = np.eye(4); M[0,0]=1-2*(yy+zz); M[0,1]=2*(xy-wz); M[0,2]=2*(xz+wy)
        M[1,0]=2*(xy+wz); M[1,1]=1-2*(xx+zz); M[1,2]=2*(yz-wx)
        M[2,0]=2*(xz-wy); M[2,1]=2*(yz+wx); M[2,2]=1-2*(xx+yy)
        return M
    def quaternion_from_matrix(M):
        R=M[:3,:3]; tr=R[0,0]+R[1,1]+R[2,2]
        if tr>0: S=np.sqrt(tr+1.0)*2; w=0.25*S; x=(R[2,1]-R[1,2])/S; y=(R[0,2]-R[2,0])/S; z=(R[1,0]-R[0,1])/S
        else:
            if R[0,0]>R[1,1] and R[0,0]>R[2,2]: S=np.sqrt(1+R[0,0]-R[1,1]-R[2,2])*2; w=(R[2,1]-R[1,2])/S; x=0.25*S; y=(R[0,1]+R[1,0])/S; z=(R[0,2]+R[2,0])/S
            elif R[1,1]>R[2,2]: S=np.sqrt(1+R[1,1]-R[0,0]-R[2,2])*2; w=(R[0,2]-R[2,0])/S; x=(R[0,1]+R[1,0])/S; y=0.25*S; z=(R[1,2]+R[2,1])/S
            else: S=np.sqrt(1+R[2,2]-R[0,0]-R[1,1])*2; w=(R[1,0]-R[0,1])/S; x=(R[0,2]+R[2,0])/S; y=(R[1,2]+R[2,1])/S; z=0.25*S
        return [x,y,z,w]
    def compose_tf(a,b):
        T1=quaternion_matrix([a[3],a[4],a[5],a[6]]); T1[:3,3]=a[:3]
        T2=quaternion_matrix([b[3],b[4],b[5],b[6]]); T2[:3,3]=b[:3]
        M=T1.dot(T2); t=M[:3,3]; q=quaternion_from_matrix(M)
        return [t[0],t[1],t[2],q[0],q[1],q[2],q[3]]

    # Normalize frame names
    pf, cf = parent_frame.strip('/'), child_frame.strip('/')

    # Read and group transforms by segment
    typestore = get_typestore(Stores.ROS1_NOETIC)
    segment_msgs = defaultdict(list)  # (h,ch) -> list of (t_ms, vals)
    with AnyReader([bagpath], default_typestore=typestore) as reader:
        conns=[c for c in reader.connections if c.topic in {topic_name,'/tf_static'}]
        for conn, ts, raw in reader.messages(connections=conns):
            msg=reader.deserialize(raw, conn.msgtype)
            t_ms=int(ts*1e-6)
            for tfm in msg.transforms:
                h=tfm.header.frame_id.strip('/'); ch=tfm.child_frame_id.strip('/')
                vals=[tfm.transform.translation.x, tfm.transform.translation.y, tfm.transform.translation.z,
                      tfm.transform.rotation.x, tfm.transform.rotation.y, tfm.transform.rotation.z, tfm.transform.rotation.w]
                segment_msgs[(h,ch)].append((t_ms, vals))
    # Sort each segment's messages once
    for seg in segment_msgs:
        segment_msgs[seg].sort(key=lambda x: x[0])

    # Build adjacency graph and find chain
    graph=defaultdict(list)
    for (h,ch) in segment_msgs:
        graph[h].append(ch)
    def find_path(s,g):
        q=deque([[s]]); seen={s}
        while q:
            p=q.popleft(); last=p[-1]
            if last==g: return p
            for nb in graph.get(last,[]):
                if nb not in seen: seen.add(nb); q.append(p+[nb])
        return None
    chain=find_path(pf,cf)
    if chain is None: raise RuntimeError(f"No route from '{pf}' to '{cf}' in {bagpath}")

    # Gather all timestamps where any segment updates
    times_set=set()
    for i in range(len(chain)-1): times_set.update([t for t,_ in segment_msgs[(chain[i],chain[i+1])]])
    tf_times=np.array(sorted(times_set),dtype=np.int64).reshape(-1,1)

    # Pre-extract time lists for bisect
    time_lists={seg:[t for t,_ in entries] for seg,entries in segment_msgs.items()}
    vals_lists={seg:[v for _,v in entries] for seg,entries in segment_msgs.items()}

    # Compose for each timestamp using binary search (O(log M) per segment)
    tf_array=[]
    for t_ms in tf_times.flatten():
        composed=None
        for i in range(len(chain)-1):
            seg=(chain[i],chain[i+1])
            times_list=time_lists[seg]; vals_list=vals_lists[seg]
            idx=bisect_right(times_list, t_ms)-1
            if idx<0: composed=None; break
            vals=vals_list[idx]
            composed=vals if composed is None else compose_tf(composed, vals)
        if composed is None:
            raise RuntimeError(f"Missing TF data for chain at t={t_ms}")
        tf_array.append(composed)

    tf_array=np.array(tf_array,dtype=np.float32)
    if verbose:
        print(f"Chain: {chain}")
        print(f"Output: times={tf_times.shape}, poses={tf_array.shape}")
    return tf_times, tf_array


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
