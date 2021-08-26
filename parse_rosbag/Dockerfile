FROM osrf/ros:melodic-desktop-full

RUN echo "source /opt/ros/melodic/setup.bash" >> ~/.bashrc

RUN apt-get update && apt-get install -y \
      python3-pip python3-yaml \
      python-catkin-tools python3-dev python3-numpy \ 
      python3-opencv \
      git && \
    rm -rf /var/lib/apt/lists/*
# CVBridge build tools

RUN pip3 install \
    rospkg catkin_pkg \
    pycryptodomex python-gnupg

 # fixes rosbag import error

WORKDIR /home/catkin_build_ws

RUN mkdir /home/catkin_build_ws/build \
        && mkdir /home/catkin_build_ws/devel \
        && mkdir /home/catkin_build_ws/install \
        && mkdir /home/catkin_build_ws/logs \
        && mkdir /home/catkin_build_ws/src \
    && catkin config \
        -DPYTHON_EXECUTABLE=/usr/bin/python3 \
        -DPYTHON_INCLUDE_DIR=/usr/include/python3.6m \
        -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.6m.so \
    && catkin config --install

WORKDIR /home/catkin_build_ws/src

RUN git clone -b melodic https://github.com/ros-perception/vision_opencv.git

WORKDIR /home/catkin_build_ws

RUN /bin/bash -c '. /opt/ros/melodic/setup.bash; catkin build cv_bridge'

RUN echo "source /home/catkin_build_ws/install/setup.bash --extend" >> ~/.bashrc

COPY src /home/src
