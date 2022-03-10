FROM nvidia/cuda:11.2.0-cudnn8-runtime-ubuntu18.04

ENV DEBIAN_FRONTEND noninteractive
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
ENV ROS_DISTRO melodic

###################################### user #####################################

ENV SHELL=/bin/bash \
    USER=medical_bed \
    UID=1000 \
    LANG=en_US.UTF-8 \
    LANGUAGE=en_US.UTF-8

ENV HOME=/home/${USER}

RUN adduser --disabled-password \
    --gecos "Default user" \
    --uid ${UID} \
    ${USER} 

RUN echo "root:root" | chpasswd
RUN echo "${USER}:111111" | chpasswd

###################################### basic tools #####################################

RUN apt-get -o Acquire::ForceIPv4=true update && apt-get -yq dist-upgrade \
    && apt-get -o Acquire::ForceIPv4=true install -yq --no-install-recommends \
    locales \
    cmake \
    make \
    git \
    vim \
    gedit \
    wget \
    sudo \
    lsb-release \
    build-essential \
    net-tools \
    dirmngr \
    gnupg2 \
    mercurial \
    libopencv-dev \
    libv4l-dev \
    libcanberra-gtk-module \
    libcanberra-gtk3-module \
    libpcap-dev \
    python3-pip \
    python3-setuptools \
    python3-opencv \
    python3-numpy \
    python3-empy  \
    python3-lxml \
    python3-pymodbus \
    terminator \
    screen \
    openssh-server \
    libglfw3-dev \
    iputils-ping \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

###################################### ROS #####################################

RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'

RUN apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654


# install bootstrap tools
RUN apt-get -o Acquire::ForceIPv4=true update && apt-get -o Acquire::ForceIPv4=true install --no-install-recommends -y \
    ros-$ROS_DISTRO-desktop-full \
    ros-$ROS_DISTRO-serial \
    ros-$ROS_DISTRO-soem \
    ros-$ROS_DISTRO-openslam-gmapping \
    ros-$ROS_DISTRO-geodesy \
    ros-$ROS_DISTRO-cartographer-* \
    ros-$ROS_DISTRO-ddynamic-reconfigure \
    ros-$ROS_DISTRO-perception \
    ros-$ROS_DISTRO-rgbd-launch \
    ros-$ROS_DISTRO-joystick-drivers \
    ros-$ROS_DISTRO-pointcloud-to-laserscan \
    ros-$ROS_DISTRO-robot-localization \
    ros-$ROS_DISTRO-spacenav-node \
    ros-$ROS_DISTRO-tf2-sensor-msgs \
    ros-$ROS_DISTRO-controller-manager \
    ros-$ROS_DISTRO-twist-mux \
    ros-$ROS_DISTRO-velodyne-simulator \
    ros-$ROS_DISTRO-gazebo-ros \
    ros-$ROS_DISTRO-move-base-msgs \
    ros-$ROS_DISTRO-lms1xx \
    ros-$ROS_DISTRO-rviz-imu-plugin \
    ros-$ROS_DISTRO-interactive-marker-twist-server \
    ros-$ROS_DISTRO-teleop-twist-joy \
    ros-$ROS_DISTRO-rosserial-arduino \
    ros-$ROS_DISTRO-rosserial \
    ros-$ROS_DISTRO-socketcan-interface \
    ros-$ROS_DISTRO-moveit-planners-ompl \
    ros-$ROS_DISTRO-moveit-* \
    ros-$ROS_DISTRO-moveit \
    ros-$ROS_DISTRO-moveit-visual-tools \
    ros-$ROS_DISTRO-velocity-controllers \
    ros-$ROS_DISTRO-joint-state-controller \
    ros-$ROS_DISTRO-joint-trajectory-controller \
    ros-$ROS_DISTRO-gazebo-ros-control \
    ros-$ROS_DISTRO-geographic-msgs \
    ros-$ROS_DISTRO-force-torque-sensor-controller \
    ros-$ROS_DISTRO-industrial-msgs \
    ros-$ROS_DISTRO-effort-controllers \
    ros-$ROS_DISTRO-nmea-msgs \
    ros-$ROS_DISTRO-rosserial-server \
    ros-$ROS_DISTRO-dynamixel-workbench-toolbox \
    ros-$ROS_DISTRO-trac-ik \
    ros-$ROS_DISTRO-vision-msgs \
    python-rosdep \
    python-rosinstall \
    python-rosinstall-generator \
    python-wstool \
    && rm -rf /var/lib/apt/lists/*

RUN rosdep init \
    && rosdep update

##################################### Serial ######################################
ENV CMAKE_PREFIX_PATH /opt/ros/$ROS_DISTRO

RUN cd ${HOME} && git clone https://github.com/wjwwood/serial.git \
    && cd serial \
    && mkdir build \
    && cd build \
    && cmake ../ -DCMAKE_INSTALL_PREFIX=${CMAKE_PREFIX_PATH} && make && make install 

######################################## realsense ########################################

RUN apt-get update && apt-get install -y \
    libsuitesparse-dev\
    software-properties-common \
    libusb-1.0-0-dev \
    libx11-dev \
    xorg-dev \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    && rm -rf /var/lib/apt/lists/*

# set the version of the realsense library
ENV LIBREALSENSE_VERSION 2.48.0
RUN cd ${HOME} && git clone https://github.com/IntelRealSense/librealsense.git \
    && cd librealsense \
    && git checkout v${LIBREALSENSE_VERSION} \
    && mkdir build \
    && cd build \
    && cmake .. \
    && make install

##################################### PIP3 ######################################

RUN pip3 install --upgrade pip setuptools

RUN pip3 install \
    pyyaml \
    rospkg \
    cython \
    catkin-tools \
    matplotlib \
    pandas \
    pypozyx \
    requests \
    scikit-image \
    scikit-learn \
    gdown \
    seaborn \
    easydict \
    lmdb \  
    shapely \
    wandb \
    virtualenv \
    pytest \
    flask \
    flask-socketio \
    loguru \
    pymongo \
    tensorflow \
    tqdm \
    modern_robotics

##################################### Pytorch ######################################

RUN python3 -m pip install -U pip 

RUN python3 -m pip install \
    torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

RUN apt-get update && apt-get install -y \
    ros-$ROS_DISTRO-smach \
    ros-$ROS_DISTRO-smach-msgs \
    ros-$ROS_DISTRO-smach-ros \
    ros-$ROS_DISTRO-executive-smach \
    ros-$ROS_DISTRO-smach-viewer

####################################### procman ###########################################

RUN cd ${HOME} && git clone https://github.com/lcm-proj/lcm \
    && cd lcm \
    && mkdir build \
    && cd build \
    && cmake .. \
    && make install

RUN cd ${HOME} && git clone https://github.com/ARG-NCTU/procman \
    && cd procman \
    && mkdir build \
    && cd build \
    && cmake .. \
    && make install

##################################### pybind11 ######################################

RUN cd ${HOME} && git clone https://github.com/pybind/pybind11.git \
    && cd pybind11 \
    && mkdir build \
    && cd build \
    && cmake ../ -DCMAKE_INSTALL_PREFIX=${CMAKE_PREFIX_PATH} && make && make install 

##################################### setting #####################################

RUN chown -R ${USER}:${USER} ${HOME}/
RUN echo "medical_bed ALL=(ALL)  ALL" > /etc/sudoers

RUN echo "source /opt/ros/noetic/setup.bash" >> ${HOME}/.bashrc

# Support of nvidia-docker 2.0
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=all

USER ${USER}

WORKDIR ${HOME}