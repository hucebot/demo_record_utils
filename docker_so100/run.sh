isRunning=`docker ps -f name=lerobot | grep -c "lerobot"`;

if [ $isRunning -eq 0 ]; then
    xhost +local:docker
    docker rm lerobot
    docker run  \
        --name lerobot  \
        --gpus all \
        -e DISPLAY=$DISPLAY \
        -e NVIDIA_DRIVER_CAPABILITIES=all \
        -e XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR \
        -v /tmp/.X11-unix:/tmp/.X11-unix \
        --env QT_X11_NO_MITSHM=1 \
        --net host \
        --privileged \
        -it \
        -v /dev:/dev \
        -v /run/udev:/run/udev \
        --device /dev/dri \
        --device /dev/snd \
        --device /dev/input \
        --device /dev/bus/usb \
        -v `pwd`/../lerobot/:/lerobot \
        -v `pwd`/../config/calibration:/lerobot/.cache/calibration \
        -v `pwd`/../config/configs.py:/lerobot/lerobot/common/robot_devices/robots/configs.py \
        -v `pwd`/copy_files.sh:/lerobot/copy_files.sh \
        -w /lerobot \
        lerobot:latest

else
    echo "Docker already running."
    docker exec -it lerobot /bin/bash
fi