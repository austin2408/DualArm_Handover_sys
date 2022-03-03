#!/usr/bin/env python3

import rospy
import smach
import smach_ros
import time
from HANet.model import HANet
from HANet.utils import processing, aff_process
import rospkg
import message_filters
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import CameraInfo, CompressedImage, Image
from geometry_msgs.msg import PoseStamped, WrenchStamped
from std_msgs.msg import Float32MultiArray
from std_srvs.srv import Trigger, TriggerResponse, TriggerRequest, SetBool, SetBoolResponse
import rospy
import os
import cv2
import torch
import copy
import numpy as np
from scipy.spatial.transform import Rotation
import warnings
from tf import TransformListener, TransformerROS, transformations
import tf
from vx300s_bringup.srv import *
warnings.filterwarnings("ignore")

arm = 'right_arm'

target = None

# define state Init
class Init(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['init'])
        self.r = TriggerRequest()

    def execute(self, userdata):
        rospy.loginfo('Executing state Init')

        # Go initial pose
        try:
            go_pose = rospy.ServiceProxy("/{0}/go_handover".format(arm), Trigger)
            resp = go_pose(self.r)
        except rospy.ServiceException as exc:
            print("service did not process request: " + str(exc))

        # open gripper
        try:
            go_pose = rospy.ServiceProxy("/{0}/gripper_open".format(arm), Trigger)
            resp = go_pose(self.r)
        except rospy.ServiceException as exc:
            print("service did not process request: " + str(exc))

        return 'init'

# Make pose prediction
class Detect(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['sucess','fail'])
        self.net = HANet(4)
        self.net.load_state_dict(torch.load(self.path+'/src/ddqn/weight/HANet.pth'))
        self.net = self.net.cuda()
        self.bridge = CvBridge()
        self.color_right = None
        self.depth_right = None
        self.r = TriggerRequest()
        self.listener = TransformListener()
        self.transformer = TransformerROS()
        info = rospy.wait_for_message('camera_right/color/camera_info', CameraInfo)
        self.fx = info.P[0]
        self.fy = info.P[5]
        self.cx = info.P[2]
        self.cy = info.P[6]

        if arm == 'right_arm':
            self.color_sub = message_filters.Subscriber('/camera_right/color/image_raw/compressed', CompressedImage)
            self.depth_sub = message_filters.Subscriber('/camera_right/aligned_depth_to_color/image_raw', Image)
        else:
            self.color_sub = message_filters.Subscriber('/camera_left/color/image_raw/compressed', CompressedImage)
            self.depth_sub = message_filters.Subscriber('/camera_left/aligned_depth_to_color/image_raw', Image)

        ts = message_filters.ApproximateTimeSynchronizer([self.color, self.depth], 5, 5)
        ts.registerCallback(self.callback_msgs)

    def callback_msgs(self, color, depth):
        self.color = color
        self.depth = depth

    def getXYZ(self, x, y, zc):
        x = float(x)
        y = float(y)
        zc = float(zc)
        inv_fx = 1.0/self.fx
        inv_fy = 1.0/self.fy
        x = (x - self.cx) * zc * inv_fx
        y = (y - self.cy) * zc * inv_fy
        z = zc

        return z, -1*x, -1*y

    def predict(self):
        # Convert msg type
        A = [90,45,0,-45]

        color = self.color
        depth = self.depth
        
        try:
            cv_image = self.bridge.compressed_imgmsg_to_cv2(color, "bgr8")
            cv_depth = self.bridge.imgmsg_to_cv2(depth, "16UC1")
            cv_depth_grasp = cv_depth.copy()
            image_pub = cv_image.copy()

        except CvBridgeError as e:
            print(e)
            return

        # Do prediction
        color_in, depth_in = processing(cv_image, cv_depth)
        color_in = color_in.cuda()
        depth_in = depth_in.cuda()

        predict = self.net(color_in, depth_in)

        predict = predict.cpu().detach().numpy()

        Max = []
        for i in range(4):
            Max.append(np.max(predict[0][i]))

        pred_id = Max.index(max(Max))

        # Get gripping point base on camera link
        x, y, aff_pub = aff_process(predict[0][pred_id], image_pub, cv_depth_grasp)

        if x != 0 and y!=0:
            z = cv_depth_grasp[int(y), int(x)]/1000.0

            aff_pub = cv2.circle(aff_pub, (int(x), int(y)), 10, (0,255,0), -1)
            p = self.bridge.cv2_to_imgmsg(aff_pub, "bgr8")
            # self.pred_img_pub.publish(p)

            camera_x, camera_y, camera_z = self.getXYZ(x, y, z)
            self.target_cam_dis = camera_z


            rot = Rotation.from_euler('xyz', [A[pred_id], 0, 0], degrees=True) 

            rot_quat = rot.as_quat()

            # Add to pose msgs
            Target_pose = ee_poseRequest()
            Target_pose.target_pose.position.x = camera_x
            Target_pose.target_pose.position.y = camera_y
            Target_pose.target_pose.position.z = camera_z

            Target_pose.target_pose.orientation.x = rot_quat[0]
            Target_pose.target_pose.orientation.y = rot_quat[1]
            Target_pose.target_pose.orientation.z = rot_quat[2]
            Target_pose.target_pose.orientation.w = rot_quat[3]

            target_pose, go_ok = self.camera2world(Target_pose)

            if z == 0.0:
                go_ok = False
            
            return target_pose, go_ok
        else:
            return None, False

    def camera2world(self, camera_pose):
        vaild = True
        try:
            if arm == 'right_arm':
                self.listener.waitForTransform('right_arm/base_link', 'camera_right_link', rospy.Time(0), rospy.Duration(1.0))
                (trans, rot) = self.listener.lookupTransform('right_arm/base_link', 'camera_right_link', rospy.Time(0))
            else:
                self.listener.waitForTransform('left_arm/base_link', 'camera_left_link', rospy.Time(0), rospy.Duration(1.0))
                (trans, rot) = self.listener.lookupTransform('left_arm/base_link', 'camera_left_link', rospy.Time(0))

        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            print("Error TF listening")
            return

        tf_pose = ee_poseRequest()

        pose = tf.transformations.quaternion_matrix(np.array(
                        [camera_pose.target_pose.orientation.x, camera_pose.target_pose.orientation.y, camera_pose.target_pose.orientation.z, camera_pose.target_pose.orientation.w]))

        pose[0, 3] = camera_pose.target_pose.position.x
        pose[1, 3] = camera_pose.target_pose.position.y
        pose[2, 3] = camera_pose.target_pose.position.z

        offset_to_world = np.matrix(transformations.quaternion_matrix(rot))
        offset_to_world[0, 3] = trans[0]
        offset_to_world[1, 3] = trans[1]
        offset_to_world[2, 3] = trans[2]

        tf_pose_matrix = np.array(np.dot(offset_to_world, pose))

        # Create a rotation object from Euler angles specifying axes of rotation
        rot = Rotation.from_matrix([[tf_pose_matrix[0, 0], tf_pose_matrix[0, 1], tf_pose_matrix[0, 2]], [tf_pose_matrix[1, 0], tf_pose_matrix[1, 1], tf_pose_matrix[1, 2]], [tf_pose_matrix[2, 0], tf_pose_matrix[2, 1], tf_pose_matrix[2, 2]]])

        # Convert to quaternions and print
        rot_quat = rot.as_quat()

        if tf_pose_matrix[0, 3] >= 0.15 and tf_pose_matrix[0, 3] <= 1.5:
            if arm == 'left_arm':
                tf_pose.target_pose.position.x = tf_pose_matrix[0, 3] + 0.07
                tf_pose.target_pose.position.y = tf_pose_matrix[1, 3] + 0.0
                tf_pose.target_pose.position.z = tf_pose_matrix[2, 3] -0.07
            else:
                tf_pose.target_pose.position.x = tf_pose_matrix[0, 3] - 0.04
                tf_pose.target_pose.position.y = tf_pose_matrix[1, 3] + 0.03
                tf_pose.target_pose.position.z = tf_pose_matrix[2, 3] - 0.1

            tf_pose.target_pose.orientation.x = rot_quat[0]
            tf_pose.target_pose.orientation.y = rot_quat[1]
            tf_pose.target_pose.orientation.z = rot_quat[2]
            tf_pose.target_pose.orientation.w = rot_quat[3]

        return tf_pose, vaild

    def execute(self, userdata):
        rospy.loginfo('Executing state Detect')
        target, GO = self.predict()

        if target == None:
            return 'fail'
        else:
            return 'sucess'

# Go target
class Move(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['move'])
        self.r = TriggerRequest()

    def execute(self, userdata):
        rospy.loginfo('Executing state Move')

        # Go pose
        try:
            go_pose = rospy.ServiceProxy("/{0}/go_pose".format(self.arm), ee_pose)
            resp = go_pose(target)
        except rospy.ServiceException as exc:
            print("service did not process request: " + str(exc))

        return 'move'


# Grasp object and move back
class Close_gripper_back(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['close'])
        self.r = TriggerRequest()

    def execute(self, userdata):
        rospy.loginfo('Executing state Grasp and Back')

        # Close gripper
        try:
            go_pose = rospy.ServiceProxy("/{0}/gripper_close".format(arm), Trigger)
            resp = go_pose(self.r)
        except rospy.ServiceException as exc:
            print("service did not process request: " + str(exc))

        rospy.sleep(0.5)

        # Back
        try:
            go_pose = rospy.ServiceProxy("/{0}/go_handover".format(arm), Trigger)
            resp = go_pose(self.r)
        except rospy.ServiceException as exc:
            print("service did not process request: " + str(exc))

        return 'close'

def main():
    # Create a SMACH state machine
    sm = smach.StateMachine(outcomes=['End'])

    with sm:
        smach.StateMachine.add('Init', Init(), transitions={'init':'Detect'})
        smach.StateMachine.add('Detect', Detect(), transitions={'sucess':'Move','fail':'Detect'})
        smach.StateMachine.add('Move', Move(), transitions={'move':'Grasp_Back'})
        smach.StateMachine.add('Grasp_Back', Close_gripper_back(), transitions={'close':'End'})

    # Create and start the introspection server
    sis = smach_ros.IntrospectionServer('my_smach_introspection_server', sm, '/SM_ROOT')
    sis.start()

    # Execute SMACH plan
    outcome = sm.execute()
    
    # Wait for ctrl-c to stop the application
    rospy.spin()
    sis.stop()

if __name__ == '__main__':
    main()
