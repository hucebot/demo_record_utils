#!/bin/bash
IsRunning=`docker ps -f name=lerobot_dataset | grep -c "lerobot_dataset"`;
if [ $IsRunning -eq "0" ]; then
    xhost +local:docker
    docker run --rm \
        --name lerobot_dataset \
        --gpus all \
        -e DISPLAY=$DISPLAY \
        -e XAUTHORITY=$XAUTHORITY \
        -e XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR \
        -e NVIDIA_DRIVER_CAPABILITIES=all \
        -e 'QT_X11_NO_MITSHM=1' \
        -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
        -v ./postprocess:/postprocess \
        -v `pwd`/../stream_deck_controller/rosbags/:/rosbags/ \
        --net host \
        --ipc host \
        --pid host \
        --device /dev/dri \
        --device /dev/snd \
        --device /dev/input \
        --device /dev/bus/usb \
        --privileged \
        --ulimit rtprio=99 \
        -ti lerobot_dataset
else
    echo "Docker image is already running. Opening new terminal...";
    docker exec -ti lerobot_dataset /bin/bash
fi