#!/usr/bin/env python3

import cv2
from cv_bridge import CvBridge

import numpy as np

import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist


class Model:
    def __init__(self, mean, std, size=(48, 64)):
        self.net = cv2.dnn.readNetFromONNX("model.onnx")

        self.size = size
        self.mean = list(mean)
        self.std = np.array(std).reshape(1, -1, 1, 1)

        self.bridge = CvBridge()
        self.twist = Twist()

        self.image_sub = rospy.Subscriber(
            "/usb_cam/image_raw", Image, self.image_callback
        )
        self.cmd_vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        self.cmd_vel_pub.publish(self.twist)

    def image_callback(self, msg):
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")

        blob = cv2.dnn.blobFromImage(image=img, size=self.size, mean=self.mean)
        blob /= self.std

        self.net.setInput(blob)
        y = self.net.forward()

        vel = y[:, 0]
        ang = y[:, 1]

        vel = 0.5 * float(1 / (1 + np.exp(-vel)))
        ang = 0.7 * float(np.tanh(ang))

        # check vel coeff is 1.2 and ang coeff is 0.7
        print(f"vel: {vel}, ang: {ang}")

        self.twist.linear.x = vel
        self.twist.angular.z = ang

        self.cmd_vel_pub.publish(self.twist)


if __name__ == "__main__":
    rospy.init_node("dnn")
    model = Model(
        mean=(6.71231037, 6.958573731, 6.694267038),
        # std=(33.4571363, 34.63760756, 34.05796006),
        std=(0.131204456, 0.135833755, 0.133560628),
        size=(48, 64),
    )
    rospy.spin()
