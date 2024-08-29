FROM nvidia/cuda:11.0-devel-ubuntu20.04

ENV TERM xterm-256color
ENV DEBIAN_FRONTEND=noninteractive

RUN rm /etc/apt/sources.list.d/cuda.list && \
    apt-key del 7fa2af80 && \
    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64/7fa2af80.pub

RUN apt update && apt install -y git \
    cmake \
    curl \
    unzip \
    build-essential \
    libboost-program-options-dev \
    libboost-filesystem-dev \
    libboost-graph-dev \
    libboost-system-dev \
    libboost-test-dev \
    libeigen3-dev \
    libsuitesparse-dev \
    libfreeimage-dev \
    libgoogle-glog-dev \
    libgflags-dev \
    libglew-dev \
    qtbase5-dev \
    libqt5opengl5-dev \
    libcgal-dev \
    libmetis-dev

RUN apt install -y libcgal-qt5-dev

# Ceres Solver
RUN apt install -y libatlas-base-dev libsuitesparse-dev libeigen3-dev

RUN commit=40c1a7e18ee040261f87b32374c9a46724ca2214 && \
    curl -LOk https://ceres-solver.googlesource.com/ceres-solver/+archive/${commit}.tar.gz && \
    mkdir -p /ceres-solver/build && \
    tar -xzf ${commit}.tar.gz -C /ceres-solver && \
    rm -f ${commit}.tar.gz && \
    cd /ceres-solver/build && \
    cmake .. -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF && \
    make -j && \
    make install

# COLMAP
RUN commit=30521f19de45c1cb2df8809728e780bf95fc8836 && \
    curl -LOk https://github.com/colmap/colmap/archive/${commit}.zip && \
    unzip ${commit}.zip -d /tmp > /dev/null && \
    rm -f ${commit}.zip && \
    mv /tmp/colmap-${commit} /colmap && \
    mkdir /colmap/build && \
    cd /colmap/build && \
    cmake .. && \
    make -j && \
    make install

# PIPELINE
RUN apt install -y python3 python3-pip
COPY ./requirements.txt /app/requirements.txt
RUN apt install -y python3 python3-pip && \
    pip install -r /app/requirements.txt
ENV PYTHONPATH /colmap/scripts/python/

# For now, comment out scripts and instead mount them in the container.
# COPY ./scripts /app
# WORKDIR /app

ENTRYPOINT ["/app/system/entrypoint.sh"]

