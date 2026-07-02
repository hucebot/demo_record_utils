FROM ros:humble

ENV TZ=Europe/Paris

# Install dependencies
RUN apt-get update && apt-get install -y \
    python3-setuptools \
    python3-pip \
    python-is-python3 \
    ffmpeg \
    libsm6 \
    libxext6 \
    ros-${ROS_DISTRO}-sensor-msgs

RUN pip install rosbags h5py opencv-python cv-bridge tyro

# Compile FFmpeg
RUN apt-get update -qq && apt-get -y install \
    autoconf \
    automake \
    build-essential \
    cmake \
    git-core \
    libass-dev \
    libfreetype6-dev \
    libgnutls28-dev \
    libmp3lame-dev \
    libsdl2-dev \
    libtool \
    libva-dev \
    libvdpau-dev \
    libvorbis-dev \
    libxcb1-dev \
    libxcb-shm0-dev \
    libxcb-xfixes0-dev \
    meson \
    ninja-build \
    pkg-config \
    texinfo \
    wget \
    yasm \
    zlib1g-dev

WORKDIR /ffmpeg_sources
WORKDIR /bin

RUN apt-get update -qq && apt-get -y install \
    libunistring-dev \
    nasm \
    libx264-dev \
    libx265-dev libnuma-dev \
    libvpx-dev \
    libfdk-aac-dev \
    libopus-dev \
    libdav1d-dev \
    libaom-dev

RUN apt-get update && apt-get install -y \
    git build-essential cmake nasm ninja-build yasm pkg-config python3 \
    libnuma-dev libaom-dev libx264-dev libx265-dev libvpx-dev libfdk-aac-dev \
    libfreetype6-dev libvorbis-dev libass-dev libssl-dev wget && \
    rm -rf /var/lib/apt/lists/*

# libsvtav1
RUN git clone --branch v2.1.0 https://gitlab.com/AOMediaCodec/SVT-AV1.git /opt/SVT-AV1 && \
    cd /opt/SVT-AV1 && mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release -G"Unix Makefiles" && \
    make -j"$(nproc)" && make install

# Update library cache so FFmpeg can find libsvtav1
RUN echo "/usr/local/lib" > /etc/ld.so.conf.d/local.conf && ldconfig

# FFmpeg
RUN git clone --depth 1 https://github.com/FFmpeg/FFmpeg.git /opt/FFmpeg && \
    cd /opt/FFmpeg && \
    PKG_CONFIG_PATH="/usr/local/lib/pkgconfig" ./configure \
      --enable-gpl \
      --enable-libsvtav1 \
      --enable-libx264 \
      --enable-libx265 \
      --enable-libvpx \
      --enable-libfdk-aac \
      --enable-libass \
      --enable-libfreetype \
      --enable-libvorbis \
      --enable-nonfree && \
    make -j"$(nproc)" && make install

RUN ffmpeg -encoders | grep svtav1 || (echo "⚠️ libsvtav1 non trouvé" && exit 1)

# Install LeRobot (tested with commit 2c22f7d76da3fcbd06ad8fbfd6baf1b13f5f0956, 25-03-2025)
WORKDIR /
RUN git clone https://github.com/huggingface/lerobot.git
WORKDIR /lerobot
RUN git checkout 790d6740babad57398e20a956a23068d377640f0
RUN pip install -e .
RUN pip install --force-reinstall -v "datasets==3.6.0"
RUN pip install matplotlib
RUN pip install streamdeck

# TODO: move these in the deps, they are here for faster build
RUN apt-get update && apt-get install -y ros-${ROS_DISTRO}-sensor-msgs \
    ros-${ROS_DISTRO}-controller-manager-msgs

RUN apt-get update && apt-get install -y \
    libhidapi-libusb0

RUN apt-get update && apt-get install -y ros-humble-rmw-cyclonedds-cpp
# --- Host User Creation ---
ARG UID
ARG GID
ARG USERNAME=data_collector

# Create group and user matching the host
RUN groupadd -g ${GID} ${USERNAME} && \
    useradd -u ${UID} -g ${GID} -m -s /bin/bash ${USERNAME}

# Ensure the new user owns the mounted/working directories
RUN mkdir -p /postprocess && chown -R ${UID}:${GID} /lerobot /postprocess

RUN pip install --no-cache-dir rerun-sdk

# Switch to the host user
USER ${USERNAME}


WORKDIR /postprocess
RUN echo "export HF_LEROBOT_HOME=/postprocess/lerobot" >> ~/.bashrc