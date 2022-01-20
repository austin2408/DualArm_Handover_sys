#!/usr/bin/env python3

import rospy
import rospkg
import os
import time
from std_srvs.srv import Trigger, TriggerResponse
from sensor_msgs.msg import Image
from sensor_msgs.msg import JointState
from std_msgs.msg import Int8
from cv_bridge import CvBridge
import message_filters
import cv2
import numpy as np
import csv


class Collect(object):
    def __init__(self, number, fps):

        # parameter
        self.number = number
        self.trigger = None
        self.traj_info = None
        self.velocity_info = None
        self.rgb = None
        self.depth = None
        self.nu = 1

        self.cv_bridge = CvBridge()
        r = rospkg.RosPack()
        self.path = os.path.join(r.get_path('vx300s_trajectory_log'), "log")

        if not os.path.exists(self.path):
            os.makedirs(self.path)

        # ros service
        start = rospy.Service("/start_collect", Trigger, self.start)
        stop = rospy.Service("/stop_collect", Trigger, self.stop)

        delay = 1/float(fps)

        # ros subscriber
        img_rgb = message_filters.Subscriber('/camera/color/image_raw', Image)
        img_depth = message_filters.Subscriber(
            '/camera/aligned_depth_to_color/image_raw', Image)
        self.ts = message_filters.ApproximateTimeSynchronizer([img_rgb, img_depth], 5, 1)
        
        self.ts.registerCallback(self.register)

        traj = rospy.Subscriber('/vx300s/joint_states',
                                JointState, self.traj_callback)

        # save data
        rospy.Timer(rospy.Duration(delay), self.save)

    def start(self, req):

        res = TriggerResponse()

        try:
            self.trigger = True
            res.success = True
            self.number += 1
            self.nu = 1
        except (rospy.ServiceException, rospy.ROSException) as e:
            res.success = False
            print("Service call failed: %s" % e)

        return res

    def stop(self, req):

        res = TriggerResponse()

        try:
            self.trigger = False
            res.success = True
        except (rospy.ServiceException, rospy.ROSException) as e:
            res.success = False
            print("Service call failed: %s" % e)

        return res

    def traj_callback(self, msg):

        self.traj_info = msg.position
        self.velocity_info = msg.velocity

    def writer_traj_csv(self, path, file_name, data, ti):

        dic = {}
        joint = ['joint0','joint1', 'joint2', 'joint3',
                 'joint4', 'joint5', 'gripper', 'left_finger', 'right_finger']
        for i in range(9):
            dic[joint[i]] = data[i]
        dic['timestamp'] = ti
        joint.append('timestamp')
        self.traj_list = []
        self.traj_list.append(dic)

        with open(os.path.join(path, file_name + '.csv'), 'a') as csvfile:

            writer = csv.DictWriter(csvfile, fieldnames=joint)
            if self.nu == 1:
                writer.writeheader()
            writer.writerows(self.traj_list)

    def writer_velocity_csv(self, path, file_name, data, ti):

        dic = {}
        vel = ['v0', 'v1', 'v2', 'v3',
                 'v4', 'v5', 'v6', 'v7', 'v8']
        for i in range(9):
            dic[vel[i]] = data[i]
        dic['timestamp'] = ti
        vel.append('timestamp')
        self.vel_list = []
        self.vel_list.append(dic)

        with open(os.path.join(path, file_name + '.csv'), 'a') as csvfile:

            writer = csv.DictWriter(csvfile, fieldnames=vel)
            if self.nu == 1:
                writer.writeheader()
            writer.writerows(self.vel_list)

    def writer_gra_csv(self, path, file_name, data, ti):

        dic = {}
        title = ['grasped_info', 'timestamp']
        dic['timestamp'] = ti
        dic['grasped_info'] = data
        self.grasped_list = []
        self.grasped_list.append(dic)

        with open(os.path.join(path, file_name + '.csv'), 'a') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=title)
            if self.nu == 1:
                writer.writeheader()
            writer.writerows(self.grasped_list)

    def register(self, rgb, depth):

        self.rgb = self.cv_bridge.imgmsg_to_cv2(rgb, "bgr8")
        self.rgb = cv2.resize(self.rgb, (224,224))
        self.depth = self.cv_bridge.imgmsg_to_cv2(depth, "16UC1")
        self.depth = cv2.resize(self.depth, (224,224))
        self.depth = np.array(self.depth) / 1000.0

    def Is_path_exists(self, path):

        if not os.path.exists(path):
            os.makedirs(path)

    def save(self, event):

        if self.trigger:

            rospy.loginfo('Start collect data!')

            log_path = os.path.join(self.path, "log_{:03}".format(self.number))
            img_path = os.path.join(log_path, "img")
            dep_path = os.path.join(log_path, "dep")

            top_img_path = os.path.join(img_path, "top")
            top_dep_path = os.path.join(dep_path, "top")

            self.Is_path_exists(log_path)
            self.Is_path_exists(img_path)
            self.Is_path_exists(dep_path)
            self.Is_path_exists(top_img_path)
            self.Is_path_exists(top_dep_path)

            ti = time.time()
            timestamp = str(ti)

            self.writer_traj_csv(log_path, "trajectory_info",
                                 self.traj_info, timestamp)
            self.writer_velocity_csv(log_path, "velocity_info",
                                self.velocity_info, timestamp)

            img_name = os.path.join(top_img_path, timestamp + "_img.jpg")
            depth_name = os.path.join(top_dep_path, timestamp + "_dep.npy")

            cv2.imwrite(img_name, self.rgb)
            np.save(depth_name, self.depth)
            self.nu += 1

if __name__ == "__main__":

    rospy.init_node("collect_data_node")

    number = rospy.get_param("number")
    fps = rospy.get_param("fps")
    collecter = Collect(number, fps)
    rospy.spin()
