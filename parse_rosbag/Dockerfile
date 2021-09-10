FROM osrf/ros:melodic-desktop-full

RUN echo "source /opt/ros/melodic/setup.bash" >> ~/.bashrc

RUN mkdir -p /home/ros_ws/src

WORKDIR /home/ros_ws

# install ros dependencies

# First get the driver:
RUN apt-get update && apt-get install -y python3-pip python3-yaml \
    python-catkin-tools python3-dev python3-numpy \
    python3-opencv \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install \
    rospkg catkin_pkg \
    pycryptodomex python-gnupg \
    pandas

WORKDIR /home/catkin_build_ws

RUN mkdir /home/catkin_build_ws/src \
        && mkdir /home/catkin_build_ws/build \
        && mkdir /home/catkin_build_ws/devel \
        && mkdir /home/catkin_build_ws/install \
        && mkdir /home/catkin_build_ws/logs \
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

WORKDIR /home/ros_ws/src

RUN catkin_create_pkg parse rospy std_msgs

COPY parse.py /home/

WORKDIR /home

# bind mount bag file directory to /home/bag_files, output directory to /home/output
# run rosrun parse dataset_from_rosbag.py bag_files output or ./parse.sh
