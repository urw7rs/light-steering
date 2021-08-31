FROM osrf/ros:melodic-desktop-full

RUN echo "source /opt/ros/melodic/setup.bash" >> ~/.bashrc

RUN mkdir -p /home/ros_ws/src

WORKDIR /home/ros_ws
# You should now be able to compile the packages using catkin

# fixes fatal error error: exploration_msgs/ExploreAction.h: No such file or directory
RUN /bin/bash -c '. /opt/ros/melodic/setup.bash; catkin_make' # --only-pkg-with-deps exploration_msgs && catkin_make -DCATKIN_WHITELIST_PACKAGES=""

RUN echo "source /home/ros_ws/devel/setup.bash" >> ~/.bashrc

WORKDIR /home/ros_ws/src

RUN catkin_create_pkg parse rospy std_msgs

COPY parse/src/* /home/ros_ws/src/parse/src/

WORKDIR /home

COPY parse.sh .

# bind mount bag file directory to /home/bag_files, output directory to /home/output
# run rosrun parse dataset_from_rosbag.py bag_files output or ./parse.sh
