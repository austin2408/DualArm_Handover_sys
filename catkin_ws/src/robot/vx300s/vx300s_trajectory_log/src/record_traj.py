#!/usr/bin/env python3

import rospy
import rospkg
import os
import time
from std_srvs.srv import Trigger, TriggerResponse, TriggerRequest
from interbotix_xs_sdk.srv import TorqueEnable, TorqueEnableRequest
from sensor_msgs.msg import Image
from sensor_msgs.msg import JointState
from std_msgs.msg import Int8
from cv_bridge import CvBridge


class Record(object):
    def __init__(self):

        # ros service
        start = rospy.Service("/start_record", Trigger, self.start_record)
        stop = rospy.Service("/stop_record", Trigger, self.stop_record)

        # ros service proxy

        self.gri_close = rospy.ServiceProxy("/vx300s/gripper_close", Trigger)
        self.gri_open = rospy.ServiceProxy("/vx300s/gripper_open", Trigger)

        self.home = rospy.ServiceProxy("/vx300s/go_mm_home", Trigger)
        self.mau = rospy.ServiceProxy("/vx300s/torque_enable", TorqueEnable)

        self.sta = rospy.ServiceProxy("/start_collect", Trigger)
        self.sto = rospy.ServiceProxy("/stop_collect", Trigger)

    def start_record(self, req):

        res = TriggerResponse()
        tor = TorqueEnableRequest()
        tri = TriggerRequest()

        tor.cmd_type = "group"
        tor.name = "all"
        tor.enable = False

        try:
            self.home(tri)
            self.sta(tri)
            time.sleep(0.5)
            self.mau(tor)
            res.success = True
        except rospy.ServiceException as exc:
            res.success = False
            print("service did not process request: " + str(exc))

        return res

    def stop_record(self, req):

        res = TriggerResponse()
        tor = TorqueEnableRequest()
        tri = TriggerRequest()

        tor.cmd_type = "group"
        tor.name = "all"
        tor.enable = True

        try:
            self.mau(tor)
            self.gri_close(tri)
            self.sto(tri)
            self.home(tri)
            time.sleep(1)
            self.gri_open(tri)
            res.success = True
        except rospy.ServiceException as exc:
            res.success = False
            print("service did not process request: " + str(exc))

        return res

if __name__ == "__main__":

    rospy.init_node("record_trajectory_node")
    record = Record()
    rospy.spin()
