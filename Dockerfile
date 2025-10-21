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
RUN sudo apt-get update -qq && sudo apt-get -y install \
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

RUN apt-get update -qq && sudo apt-get -y install \
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

# libsvtav1
#WORKDIR /ffmpeg_sources
#RUN git clone https://gitlab.com/AOMediaCodec/SVT-AV1.git
#WORKDIR /ffmpeg_sources/SVT-AV1/build
#RUN PATH="$HOME/bin:$PATH" cmake -G "Unix Makefiles" -DCMAKE_INSTALL_PREFIX="$HOME/ffmpeg_build" -DCMAKE_BUILD_TYPE=Release -DBUILD_DEC=OFF -DBUILD_SHARED_LIBS=OFF .. && \
#    PATH="$HOME/bin:$PATH" make && \ 
#    make install

# libaom
#WORKDIR /ffmpeg_sources
#RUN git clone --depth 1 https://aomedia.googlesource.com/aom
#WORKDIR /ffmpeg_sources/aom_build
#RUN PATH="$HOME/bin:$PATH" cmake -G "Unix Makefiles" -DCMAKE_INSTALL_PREFIX="$HOME/ffmpeg_build" -DENABLE_TESTS=OFF -DENABLE_NASM=on ../aom && \
#    PATH="$HOME/bin:$PATH" make && \
#    make install

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

#WORKDIR /ffmpeg_sources
#RUN wget -O ffmpeg-snapshot.tar.bz2 https://ffmpeg.org/releases/ffmpeg-snapshot.tar.bz2 && \
#    tar xjvf ffmpeg-snapshot.tar.bz2
#WORKDIR /ffmpeg_sources/ffmpeg
#RUN PATH="$HOME/bin:$PATH" PKG_CONFIG_PATH="$HOME/ffmpeg_build/lib/pkgconfig" ./configure \
#    --prefix="$HOME/ffmpeg_build" \
#    --pkg-config-flags="--static" \
#    --extra-cflags="-I$HOME/ffmpeg_build/include" \
#    --extra-ldflags="-L$HOME/ffmpeg_build/lib" \
#    --extra-libs="-lpthread -lm" \
#    --ld="g++" \
#    --bindir="$HOME/bin" \
#    --enable-gpl \
#    --enable-gnutls \
#    --enable-libaom \
#    --enable-libass \
#    --enable-libfdk-aac \
#    --enable-libfreetype \
#    --enable-libmp3lame \
#    --enable-libopus \
#    --enable-libsvtav1 \
#    --enable-libdav1d \
#    --enable-libvorbis \
#    --enable-libvpx \
#    --enable-libx264 \
#    --enable-libx265 \
#    --enable-nonfree 
#RUN PATH="$HOME/bin:$PATH" make && make install
#RUN cp ~/bin/ffmpeg /usr/bin/ffmpeg


#RUN apt-get update && apt-get install -y --no-install-recommends \
#    software-properties-common build-essential git curl \
#    libglib2.0-0 libgl1-mesa-glx libegl1-mesa ffmpeg \
#    libusb-1.0-0-dev speech-dispatcher libgeos-dev portaudio19-dev \
#    cmake pkg-config ninja-build \

RUN ffmpeg -encoders | grep svtav1 || (echo "⚠️ libsvtav1 non trouvé" && exit 1)

#CMD ["ffmpeg", "-version"]

# Install LeRobot (tested with commit 2c22f7d76da3fcbd06ad8fbfd6baf1b13f5f0956, 25-03-2025)
WORKDIR /
RUN git clone https://github.com/huggingface/lerobot.git
WORKDIR /lerobot
RUN git checkout 790d6740babad57398e20a956a23068d377640f0
RUN pip install -e .
RUN pip install --force-reinstall -v "datasets==3.6.0"
RUN pip install matplotlib

WORKDIR /postprocess
RUN echo "export HF_LEROBOT_HOME=/postprocess/lerobot" >> ~/.bashrc

