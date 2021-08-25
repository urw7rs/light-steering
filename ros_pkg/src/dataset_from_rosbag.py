#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2016 Massachusetts Institute of Technology

import os

import argparse
import csv

import cv2

import rosbag
from cv_bridge import CvBridge


def get_label(msg):
    vel = msg.linear.x
    ang = msg.angular.z

    return vel, ang


def main():
    parser = argparse.ArgumentParser(
        description="Extract images and labels from a ROS bag"
    )
    parser.add_argument("bag_file", help="ros bag file")
    parser.add_argument("output_dir", help="directory to save images and labels")
    parser.add_argument("image_topic", default="/usb_cam/image_raw", help="image topic")
    parser.add_argument("label_topic", default="/cmd_vel", help="label topic")

    args = parser.parse_args()

    bag = rosbag.Bag(args.bag_file, "r")
    bridge = CvBridge()

    os.mkdir(os.path.join(args.output_dir, "images"))

    # map labels to images
    #
    # labels and images aren't synchronized
    # map the latest label to an image
    # if no labels are present between images, use the previous label
    path_list = []
    label_list = []
    n_images = 0
    n_labels = 0
    copied = 0
    for topic, msg, t in bag.read_messages(topics=[args.image_topic, args.label_topic]):
        if topic == args.image_topic:
            n_images += 1

            cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

            path = os.path.join(args.output_dir, "images", f"{n_images}.png")
            cv2.imwrite(path, cv_img)

            path_list.append(path)

            if len(label_list) - n_images > 1:
                label_list.append(label_list[-1])
                copied += 1

        else:
            n_labels += 1

            label = get_label(msg)

            if n_images > len(label_list):
                label_list.append(label)
            elif n_images == len(label_list):
                # overwrite only when there's useful labels
                if label[1] != 0.0:
                    label_list[n_images - 1] = label

    bag.close()

    print(
        f"{n_images} images written, {n_labels} labels recieved, {copied} labels copied over"
    )

    csv_path = os.path.join(args.output_dir, "label.csv")
    with open(csv_path, "w", newline="") as csvfile:
        csv_writer = csv.writer(
            csvfile, delimiter=" ", quotechar="|", quoting=csv.QUOTE_MINIMAL
        )

        for path, label in zip(path_list, label_list):
            vel, ang = label
            csv_writer.writerow([path, vel, ang])


if __name__ == "__main__":
    main()
