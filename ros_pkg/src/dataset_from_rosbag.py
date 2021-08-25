#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2016 Massachusetts Institute of Technology

import os

import argparse

import cv2

import rosbag
from cv_bridge import CvBridge


def get_label(msg):
    vel = msg.linear.x
    ang = msg.angular.z

    return vel, ang


def main():
    """Extract a folder of images and csv label file from a rosbag."""
    parser = argparse.ArgumentParser(
        description="Extract images and labels from a ROS bag."
    )
    parser.add_argument("bag_file", help="Input ROS bag.")
    parser.add_argument("output_dir", help="Output directory.")
    parser.add_argument("image_topic", help="Image topic.")
    parser.add_argument("label_topic", nargs="+", help="Image topic.")

    args = parser.parse_args()

    print(
        "Extract images from %s on topic %s into %s"
        % (args.bag_file, args.image_topic, args.output_dir)
    )

    bag = rosbag.Bag(args.bag_file, "r")
    bridge = CvBridge()

    parsed_images = {"path": [], "t": []}

    i = 0
    for topic, msg, t in bag.read_messages(
        topics=[args.image_topic, *args.label_topic]
    ):
        cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

        path = os.path.join(args.output_dir, f"{i}.png")
        cv2.imwrite(path, cv_img)

        parsed_images["path"].append(path)
        parsed_images["t"].append(t.to_time())

        i += 1

    base_dict = {"label": [], "t": []}
    parsed_labels = {k: base_dict.copy() for k in args.label_topic}
    i = 0
    for topic, msg, t in bag.read_messages(topics=args.label_topic):
        current_dict = parsed_labels[topic]

        label = get_label(msg)

        print(f"labeled image {i}")

        i += 1

        current_dict["label"].append(label)
        current_dict["t"].append(t.to_time())

    bag.close()


if __name__ == "__main__":
    main()
