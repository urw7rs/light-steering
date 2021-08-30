import os
import unittest

import csv

import rosbag

from src import dataset_from_rosbag as BagParser

from cv_bridge import CvBridge

import numpy as np

from geometry_msgs.msg import Twist, Vector3


class TestBagParser(unittest.TestCase):
    def test_parsed_label(self):
        vel_test = 0.7
        ang_test = 0.3

        cmd_vel = Twist()
        cmd_vel.linear = Vector3(vel_test, 0.0, 0.0)
        cmd_vel.angular = Vector3(0.0, 0.0, ang_test)

        parser = BagParser.get_label

        vel, ang = parser(cmd_vel)

        self.assertEqual(vel, vel_test, f"Should be {vel_test}")
        self.assertEqual(ang, ang_test, f"Should be {ang_test}")

    def test_topic_name(self):
        image_topic = "/usb_cam/image_raw"
        label_topic = "/cmd_vel"

        self.assertEqual(BagParser.image_topic, image_topic, f"Should be {image_topic}")
        self.assertEqual(BagParser.label_topic, label_topic, f"Should be {label_topic}")

    def _write_image(self, bag):
        img = np.zeros((10, 10, 1), np.uint8)
        bridge = CvBridge()

        msg = bridge.cv2_to_imgmsg(img, encoding="passthrough")

        bag.write(BagParser.image_topic, msg)

    def _write_label(self, bag):
        vel_test = 0.7
        ang_test = 0.3

        cmd_vel = Twist()
        cmd_vel.linear = Vector3(vel_test, 0.0, 0.0)
        cmd_vel.angular = Vector3(0.0, 0.0, ang_test)

        bag.write(BagParser.label_topic, cmd_vel)

    def test_file_parse(self):
        bag_path = "test.bag"
        output_path = "test_output"

        bag = rosbag.Bag(bag_path, "w")

        self._write_image(bag)

        os.mkdir(output_path)

        bridge = CvBridge()

        path_list, label_list = BagParser.parse_bag(bag, bridge, output_path)

        bag.close()

        path_dict = {k: label for k, label in zip(path_list, label_list)}

        existing_paths = os.listdir(os.path.join(output_path, "image"))
        for i, path in enumerate(existing_paths):
            existing_paths[i] = os.path.join(output_path, "image", path)

        self.assertIn("label.csv", os.listdir(output_path), "label.csv not written")

        with open(os.path.join(output_path, "label.csv"), "r") as csvfile:
            csv_reader = csv.reader(csvfile, deliminator=",", quotechar="|")

            for path, vel, ang in csv_reader:
                self.assertIn(path, existing_paths, "{path} wasn't written")

                vel_test, ang_test = path_dict[path]
                self.assertEqual(vel, vel_test)
                self.assertEqual(ang, ang_test)


if __name__ == "__main__":
    unittest.main()
