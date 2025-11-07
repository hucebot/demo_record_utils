#!/bin/bash

# usage
# ./record.sh --demo_task1
# ./record.sh --clean demo_task1
  
# === CONFIGURATION ===
BASE_DIR="./"   # current directory (where the script is located)

# === CLEAN SECTION ===
if [ "$1" = "--clean" ]; then
  DEMO=$2
  if [ -z "$DEMO" ]; then
    echo "Please specify which demo to clean, e.g.: ./record.sh --clean demo_task1"
    exit 1
  fi
  TARGET_DIR="$BASE_DIR/$DEMO"
  if [ ! -d "$TARGET_DIR" ]; then
    echo "No folder found for $DEMO"
    exit 1
  fi

  cd "$TARGET_DIR" || exit 1
  LAST_BAG=$(ls -t | head -n 1)
  if [ -n "$LAST_BAG" ]; then
    echo "Removing latest bag: $LAST_BAG"
    rm -rf "$LAST_BAG"
    echo "Successfully deleted."
  else
    echo "No bag found in $TARGET_DIR"
  fi
  exit 0
fi

# === RECORD SECTION ===
if [[ "$1" == --* ]]; then
  DEMO=${1/--/}   # remove the double dashes
else
  echo "Usage:"
  echo "   ./record.sh --demo_task1"
  echo "   ./record.sh --clean demo_task1"
  exit 1
fi

# create demo folder if not existing
SAVE_DIR="$BASE_DIR/$DEMO"
mkdir -p "$SAVE_DIR"

BAG_NAME="${DEMO}_$(date +%Y%m%d_%H%M%S)"

# === TOPIC LIST ===
TOPICS=(
  /camera/camera/color/camera_info
  /camera/camera/color/image_raw/compressed
  /camera/camera/depth/camera_info
  /camera/camera/depth/image_rect_raw
  /webcam1/image_raw/compressed
  /webcam2/image_raw/compressed
  /webcam3/image_raw/compressed
  /cartesian_impedance/cartesian_pos_curr
  /cartesian_impedance/cartesian_pos_des_filt
  /cartesian_impedance/equilibrium_pose
  /cartesian_impedance/f_ext_cart
  /cartesian_impedance/joint_state
  /panda_gripper/gripper_command
  /panda_gripper/width
)

cd "$SAVE_DIR" || exit 1
ros2 bag record -o "$BAG_NAME" "${TOPICS[@]}" -b 2000000000

