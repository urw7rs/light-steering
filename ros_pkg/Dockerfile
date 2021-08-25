FROM osrf/ros:melodic-desktop-full

RUN apt-get update && apt-get install -y \
      ros-${ROS_DISTRO}-cv-bridge \
      ros-${ROS_DISTRO}-vision-opencv \
      python3-pip python3-yaml \
      python-catkin-tools python3-dev \
      python3-numpy git && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install rospkg catkin-pkg jupyterlab

RUN catkin config \
      -DPYTHON_EXECUTABLE=/usr/bin/python3 \
      -DPYTHON_INCLUDE_DIR=/usr/include/python3.6m \
      -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.6m.so && \
    catkin config --install

RUN mkdir -p /src

WORKDIR /src

RUN git clone -b melodic https://github.com/ros-perception/vision_opencv.git

WORKDIR /

RUN catkin build cv_bridge && \
    source install/setup.bash --extend

WORKDIR /src

RUN git clone https://github.com/ros/ros 

WORKDIR /

RUN catkin_make_isolated --install --pkg rosunit -DCMAKE_BUILD_TYPE=Release --install-space /opt/ros/melodic
