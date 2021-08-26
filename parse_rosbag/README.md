# code for generating datasets from rosbag files

## usage

build the image

`docker build -t parsebag .`

## testing

change your working directory to docker-tests.

build and run container to run the test scripts

**the base container in the parent directory needs to be tagged as parsebag**

`docker build -t parsebag:test . && docker run --rm parsebag:test`
