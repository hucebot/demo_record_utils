"""
How one message becomes numbers or an image: shared by the rosbag conversion (utils.py, messages deserialised by
rosbags) and live consumers such as ForceVAM's policy node (rclpy messages: the same field names), so a policy sees
at run time exactly what it was trained on. numpy and OpenCV only.
"""
import cv2
import numpy as np

IMAGE_TYPES = ("sensor_msgs/msg/CompressedImage", "sensor_msgs/msg/Image")
NUMERIC_TYPES = ("sensor_msgs/msg/JointState", "geometry_msgs/msg/PoseStamped", "geometry_msgs/msg/WrenchStamped",
                 "custom_msgs/msg/GripperWidth", "geometry_msgs/msg/PointStamped")


def stamp_seconds(msg):
    """The header stamp in s (0.0 when the message has no header or an unset stamp)."""
    if not hasattr(msg, "header"):
        return 0.0
    return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9


def decode_compressed(data):
    """A CompressedImage's data buffer to an OpenCV image as stored (BGR for color, 16-bit PNG depth intact)."""
    return cv2.imdecode(np.frombuffer(bytes(data), dtype=np.uint8), cv2.IMREAD_UNCHANGED)


def raw_image_to_array(msg):
    """sensor_msgs/Image to an array (H, W[, C]), RGB for color, without cv_bridge."""
    dtype = {"mono16": np.uint16, "16UC1": np.uint16, "32FC1": np.float32}.get(msg.encoding, np.uint8)
    channels = {"rgb8": 3, "bgr8": 3, "rgba8": 4, "bgra8": 4}.get(msg.encoding, 1)
    row = np.frombuffer(bytes(msg.data), dtype=dtype).reshape(msg.height, -1)[:, :msg.width * channels]
    img = row.reshape(msg.height, msg.width, channels) if channels > 1 else row.reshape(msg.height, msg.width)
    if msg.encoding in ("bgr8", "bgra8"):
        img = img[..., [2, 1, 0] + ([3] if channels == 4 else [])]
    return img


def image_array(msg_or_data, msg_type, image_size):
    """
    A camera message (or a CompressedImage's data buffer) to the stored image: RGB, resized to image_size
    (width, height) with INTER_AREA, (H, W, C) uint8 (a channel axis added to single-channel images).
    """
    if msg_type == "sensor_msgs/msg/CompressedImage":
        data = msg_or_data.data if hasattr(msg_or_data, "data") else msg_or_data
        img = decode_compressed(data)
        if img.ndim == 3 and img.shape[-1] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img = raw_image_to_array(msg_or_data)
    img = cv2.resize(img, tuple(image_size), interpolation=cv2.INTER_AREA)
    return img[..., None] if img.ndim == 2 else img


def numeric_values(msg, msg_type, args):
    """The numbers one field takes from a numeric message (args: the field's config, e.g. physical_quantity)."""
    if msg_type == "sensor_msgs/msg/JointState":
        return list(getattr(msg, args.get("physical_quantity", "position")))
    if msg_type == "geometry_msgs/msg/PoseStamped":  # position, then the quaternion (x, y, z, w)
        p, q = msg.pose.position, msg.pose.orientation
        return [p.x, p.y, p.z, q.x, q.y, q.z, q.w]
    if msg_type == "geometry_msgs/msg/WrenchStamped":
        quantity = args.get("physical_quantity")
        if quantity not in ("force", "torque"):
            raise ValueError(f"WrenchStamped fields need physical_quantity: force or torque (got {quantity})")
        v = getattr(msg.wrench, quantity)
        return [v.x, v.y, v.z]
    if msg_type == "custom_msgs/msg/GripperWidth":
        return [msg.width]
    if msg_type == "geometry_msgs/msg/PointStamped":
        return [msg.point.x]
    raise NotImplementedError(msg_type)
