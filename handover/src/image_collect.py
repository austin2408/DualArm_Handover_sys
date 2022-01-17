#!/usr/bin/env python3
import rospkg
import message_filters
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import CameraInfo, CompressedImage, Image
from std_srvs.srv import Trigger, TriggerResponse, TriggerRequest, SetBool, SetBoolResponse
import rospy
import os
import cv2
import numpy as np
import warnings
warnings.filterwarnings("ignore")


class Collect():
    def __init__(self):
        self.bridge = CvBridge()
        r = rospkg.RosPack()
        self.path = r.get_path("handover")

        # global variable
        self.colorMsg_right = None
        self.depthMsg_right = None
        self.colorMsg_left = None
        self.depthMsg_left = None
        self.num = 587
        self.arm = 'left_arm'
        info = rospy.wait_for_message('camera_right/color/camera_info', CameraInfo)


        # Mssage filter
        self.color_right = message_filters.Subscriber('/camera_right/color/image_raw/compressed', CompressedImage)
        self.depth_right = message_filters.Subscriber('/camera_right/aligned_depth_to_color/image_raw', Image)

        self.color_left = message_filters.Subscriber('/camera_left/color/image_raw/compressed', CompressedImage)
        self.depth_left = message_filters.Subscriber('/camera_left/aligned_depth_to_color/image_raw', Image)

        ts = message_filters.ApproximateTimeSynchronizer([self.color_right, self.depth_right, self.color_left, self.depth_left], 5, 5)
        ts.registerCallback(self.callback_msgs)

        # Service
        rospy.Service('~collect', Trigger, self.strategy)
        # update_server = rospy.Service('~update_server', SetBool, self.update)
        self.reset_arm()

    def reset_arm(self):
        r = TriggerRequest()

        try:
            go_initial = rospy.ServiceProxy("/{0}/go_handover".format('right_arm'), Trigger)
            resp = go_initial(r)

            go_initial = rospy.ServiceProxy("/{0}/go_handover".format('left_arm'), Trigger)
            resp = go_initial(r)

            # res.success = True
        except rospy.ServiceException as exc:
            # res.success = False
            print("service did not process request: " + str(exc))

        rospy.loginfo('Reset arms to initial state')


    def callback_msgs(self, colorR, depthR, colorL, depthL):
        self.color_right = colorR
        self.depth_right = depthR

        self.color_left = colorL
        self.depth_left = depthL

    def strategy(self, req):
        res = TriggerResponse()

        r = TriggerRequest()

        # Make prediction
        rospy.loginfo('============================================')
        if self.arm == 'right_arm':
            rospy.loginfo('right_arm')
            try:
                cv_image = self.bridge.compressed_imgmsg_to_cv2(self.color_right, "bgr8")
                cv_depth = self.bridge.imgmsg_to_cv2(self.depth_right, "16UC1")

            except CvBridgeError as e:
                print(e)
                return
        else:
            rospy.loginfo('left_arm')
            try:
                cv_image = self.bridge.compressed_imgmsg_to_cv2(self.color_left, "bgr8")
                cv_depth = self.bridge.imgmsg_to_cv2(self.depth_left, "16UC1")

            except CvBridgeError as e:
                print(e)
                return

        cv2.imwrite(self.path+'/src/colornew/color_'+str(self.num)+'.jpg', cv_image)
        np.save(self.path+'/src/depthnew/depth_'+str(self.num), cv_depth)

        self.num += 1

        if self.arm == 'right_arm':
            self.arm = 'left_arm'
        else:
            self.arm = 'right_arm'

        return res

    def onShutdown(self):
        r = TriggerRequest()
        try:
            go_initial = rospy.ServiceProxy("/{0}/go_rise".format('right_arm'), Trigger)
            resp = go_initial(r)

            go_initial = rospy.ServiceProxy("/{0}/go_rise".format('left_arm'), Trigger)
            resp = go_initial(r)

            # res.success = True
        except rospy.ServiceException as exc:
            # res.success = False
            print("service did not process request: " + str(exc))

        try:
            go_initial = rospy.ServiceProxy("/{0}/go_sleep".format('right_arm'), Trigger)
            resp = go_initial(r)

            go_initial = rospy.ServiceProxy("/{0}/go_sleep".format('left_arm'), Trigger)
            resp = go_initial(r)

            # res.success = True
        except rospy.ServiceException as exc:
            # res.success = False
            print("service did not process request: " + str(exc))

        rospy.loginfo("Shutdown.")



if __name__ == '__main__':
    rospy.init_node("collect", anonymous=False)
    collect = Collect()
    rospy.on_shutdown(collect.onShutdown)
    rospy.spin()
