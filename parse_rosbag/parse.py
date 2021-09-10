#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2016 Massachusetts Institute of Technology

import os

from multiprocessing import Pool

import argparse

import numpy as np
import pandas as pd

import cv2

import rosbag
from cv_bridge import CvBridge


def get_label(msg):
    vel = msg.linear.x
    ang = msg.angular.z

    return vel, ang


def parse_bag(bag_path):
    bag = rosbag.Bag(os.path.join(bagdir_path, bag_path), "r")

    sub_path = os.path.join(output_path, bag_path[:-4])
    if not os.path.isdir(sub_path):
        os.mkdir(sub_path)

    img_path = os.path.join(sub_path, "image")
    if not os.path.isdir(img_path):
        os.mkdir(img_path)

    bridge = CvBridge()

    # map labels to images
    #
    # labels and images aren't synchronized
    # map the latest label to an image
    # if no labels are present between images, use the previous label
    path_list = []
    label_list = []
    n_image = 0
    n_labels = 0
    copied = 0
    for topic, msg, t in bag.read_messages(topics=[image_topic, label_topic]):
        if topic == image_topic:
            n_image += 1

            cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

            path = os.path.join("image", f"{n_image}.png")
            save_path = os.path.join(sub_path, path)
            if os.path.isfile(save_path):
                print(f"skipping {save_path} already written")
            else:
                cv2.resize(cv_img, tuple(img_size))
                cv2.imwrite(save_path, cv_img)

            path_list.append(path)

            if n_image - len(label_list) > 1:
                label_list.append(label_list[-1])
                copied += 1

        else:
            n_labels += 1

            label = get_label(msg)

            if n_image > len(label_list):
                label_list.append(label)
            elif n_image == len(label_list):
                # overwrite only when there's useful labels
                if label[1] != 0.0:
                    label_list[n_image - 1] = label

    if n_image > len(label_list):
        label_list.append(label_list[-1])
        copied += 1

    label = np.array(label_list)
    path = np.array(path_list).reshape(-1, 1)

    data = np.concatenate((path, label), axis=1)

    df = pd.DataFrame(data, columns=["path", "vel", "ang"])

    csv_path = os.path.join(output_path, "label.csv")
    df.to_csv(csv_path, index=False)

    bag.close()

    print(f"parsed {bag_path}: {n_image} written, {copied} copied ")


image_topic = "/usb_cam/image_raw"
label_topic = "/cmd_vel"


parser = argparse.ArgumentParser()
parser.add_argument("bagdir_path", help="ros bag file")
parser.add_argument(
    "output_path",
    help="directory to save images and labels",
)
parser.add_argument(
    "--img_size",
    type=int,
    nargs=2,
    default=[48, 64],
    help="directory to save images and labels",
)

args = parser.parse_args()

bagdir_path = args.bagdir_path
output_path = args.output_path
img_size = tuple(args.img_size)

if __name__ == "__main__":
    with Pool(len(os.sched_getaffinity(0))) as p:
        p.map(parse_bag, os.listdir(args.bagdir_path))
