#!/bin/bash

SESSION="data_collection_franka"
TASK_NAME=$1

# 1. Check if a task argument was provided
if [ -z "$TASK_NAME" ]; then
  echo "Error: No task provided."
  echo "Usage: ./begin_teleop_franka.sh <task_name>"
  exit 1
fi

# 2. Check if the session already exists
tmux has-session -t $SESSION 2>/dev/null

if [ $? != 0 ]; then
  # 3. If it does NOT exist, create the session and the layout
  tmux new-session -d -s $SESSION

  # Pane 1 (Left): Motion Recorder
  tmux send-keys -t $SESSION "cd ~/code/demo_record_utils_dionisis && make run && make record ROBOT=franka TASK=$TASK_NAME" C-m

  # Pane 2 (Top Right): Vive Controllers
  tmux split-window -h -t $SESSION
  tmux send-keys -t $SESSION "cd ~/code/vive_controller_dionisis && make franka" C-m

  # Pane 3 (Bottom Right): Cameras
  tmux split-window -v -t $SESSION
  tmux send-keys -t $SESSION "cd ~/code/docker_cams && docker compose up" C-m
else
  echo "Session '$SESSION' already running. Attaching instead of creating new panes."
fi

# 4. Attach safely (prevents nesting errors if you are already inside tmux)
if [ -z "$TMUX" ]; then
  tmux attach-session -t $SESSION
else
  tmux switch-client -t $SESSION
fi