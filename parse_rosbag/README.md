# code for generating datasets from rosbag files

## usage

build the image

needs ros_ws but not included in the repository (too big)

`docker build -t parsebag .`

bind mount bag file directory to /home/bag_files, output directory to /home/output


run `rosrun parse dataset_from_rosbag.py bag_files output`

or run `./parse.sh`

if there are errors about cv_bridge run `source devel/setup.bash` and `catkin_ws/install/setup.bash --extend`

## testing

### automatic
running test.sh will run the steps below for you. run in the same directory as the script

### manual
change your working directory to docker-tests.

build and run container to run the test scripts

**the base container in the parent directory needs to be tagged as parsebag**

`docker build -t parsebag:test . && docker run --rm parsebag:test`
