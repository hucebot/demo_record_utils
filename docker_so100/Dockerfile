FROM nvidia/cuda:12.4.1-base-ubuntu22.04

###### Set environment variable
ENV DISPLAY=:0
ENV ROS_DISTRO=noetic
ENV LIBGL_ALWAYS_INDIRECT=0
ARG PYTHON_VERSION=3.10
ENV DEBIAN_FRONTEND=noninteractive
ENV MUJOCO_GL="egl"
ENV PATH="/opt/venv/bin:$PATH"

# Configure environment variables
ARG PYTHON_VERSION=3.10
ENV DEBIAN_FRONTEND=noninteractive
ENV MUJOCO_GL="egl"
ENV PATH="/opt/venv/bin:$PATH"

ENV DEBIAN_FRONTEND="noninteractive"
ENV TZ="UTC"

###### Install basic dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git \
    libglib2.0-0 libgl1-mesa-glx libegl1-mesa ffmpeg \
    speech-dispatcher libgeos-dev \
    python${PYTHON_VERSION}-dev \
    python-is-python3 \
    python3-pip \
    terminator

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git \
    libglib2.0-0 libgl1-mesa-glx libegl1-mesa ffmpeg \
    speech-dispatcher libgeos-dev

RUN apt-get install -y --reinstall \
    libqt5core5a libqt5gui5 libqt5widgets5 \
    libxcb-xinerama0 libxkbcommon-x11-0 \
    libqt5x11extras5

RUN apt-get install speech-dispatcher

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
    libdav1d-dev

###### Install and compile libsvtav1
WORKDIR /ffmpeg_sources
RUN git clone https://gitlab.com/AOMediaCodec/SVT-AV1.git
WORKDIR /ffmpeg_sources/SVT-AV1/build
RUN PATH="$HOME/bin:$PATH" cmake -G "Unix Makefiles" -DCMAKE_INSTALL_PREFIX="$HOME/ffmpeg_build" -DCMAKE_BUILD_TYPE=Release -DBUILD_DEC=OFF -DBUILD_SHARED_LIBS=OFF .. && \
    PATH="$HOME/bin:$PATH" make && \ 
    make install

###### Install and compile libaom
WORKDIR /ffmpeg_sources
RUN git clone --depth 1 https://aomedia.googlesource.com/aom
WORKDIR /ffmpeg_sources/aom_build
RUN PATH="$HOME/bin:$PATH" cmake -G "Unix Makefiles" -DCMAKE_INSTALL_PREFIX="$HOME/ffmpeg_build" -DENABLE_TESTS=OFF -DENABLE_NASM=on ../aom && \
    PATH="$HOME/bin:$PATH" make && \
    make install

###### Install and compile FFmpeg
WORKDIR /ffmpeg_sources
RUN wget -O ffmpeg-snapshot.tar.bz2 https://ffmpeg.org/releases/ffmpeg-snapshot.tar.bz2 && \
    tar xjvf ffmpeg-snapshot.tar.bz2
WORKDIR /ffmpeg_sources/ffmpeg
RUN PATH="$HOME/bin:$PATH" PKG_CONFIG_PATH="$HOME/ffmpeg_build/lib/pkgconfig" ./configure \
    --prefix="$HOME/ffmpeg_build" \
    --pkg-config-flags="--static" \
    --extra-cflags="-I$HOME/ffmpeg_build/include" \
    --extra-ldflags="-L$HOME/ffmpeg_build/lib" \
    --extra-libs="-lpthread -lm" \
    --ld="g++" \
    --bindir="$HOME/bin" \
    --enable-gpl \
    --enable-gnutls \
    --enable-libaom \
    --enable-libass \
    --enable-libfdk-aac \
    --enable-libfreetype \
    --enable-libmp3lame \
    --enable-libopus \
    --enable-libsvtav1 \
    --enable-libdav1d \
    --enable-libvorbis \
    --enable-libvpx \
    --enable-libx264 \
    --enable-libx265 \
    --enable-nonfree 
RUN PATH="$HOME/bin:$PATH" make && make install
RUN cp ~/bin/ffmpeg /usr/bin/ffmpeg

###### Install LeRobot - Tested with commit: 78fd1a1e04f87f3ec52428f72c29485124472fbc, 27-03-2025
RUN git clone https://github.com/huggingface/lerobot.git /lerobot
RUN cd /lerobot && pip install -e ".[feetech]"

###### Install python packages 
RUN pip install pyserial \
    lerobot


