#!/usr/bin/env python3

import rospy
from std_srvs.srv import Trigger, TriggerResponse
from interbotix_xs_modules.arm import InterbotixManipulatorXS
from vx300s_bringup.srv import *
from sensor_msgs.msg import JointState

import tf
import tf.transformations as tfm

class vx300s():
    def __init__(self, name):

        self.name = name

        # Service
        rospy.Service("/{0}/go_home".format(name), Trigger, self.vx300s_home)
        rospy.Service("/{0}/go_sleep".format(name), Trigger, self.vx300s_sleep)
        rospy.Service("/{0}/go_pose".format(name), ee_pose, self.vx300s_ee_pose)
        rospy.Service("/{0}/gripper_open".format(name), Trigger, self.vx300s_open)
        rospy.Service("/{0}/gripper_close".format(name), Trigger, self.vx300s_close)
        rospy.Service("/{0}/check_grasped".format(name), Trigger, self.vx300s_check)
        rospy.Service("/{0}/go_mm_home".format(name), Trigger, self.mm_home)
        rospy.Service("/{0}/rotation".format(name), Trigger, self.rot)
        rospy.Service("/{0}/get_pose".format(name), cur_pose, self.get_pose)

        self.listener = tf.TransformListener()

        # vx300s setup
        robot = InterbotixManipulatorXS(robot_model="vx300s", group_name="arm", gripper_name="gripper", robot_name=name, init_node=False)

        self.arm = robot.arm
        self.gripper = robot.gripper

        self.init()

    def init(self):

        self.gripper.open(2.0)
        self.arm.go_to_sleep_pose()
        rospy.loginfo("initial already!")

    def vx300s_home(self, req):

        res = TriggerResponse()

        try:
            self.arm.go_to_home_pose()
            res.success = True
        except (rospy.ServiceException, rospy.ROSException) as e:
            res.success = False
            print("Service call failed: %s"%e)
        
        return res

    def vx300s_sleep(self, req):

        res = TriggerResponse()

        try:
            self.arm.go_to_sleep_pose()
            res.success = True
        except (rospy.ServiceException, rospy.ROSException) as e:
            res.success = False
            print("Service call failed: %s"%e)
        
        return res

    def vx300s_check(self, req):
        
        res = TriggerResponse()

        try:
            joint_info = rospy.wait_for_message('/{0}/joint_states'.format(self.name), JointState)
        except (rospy.ServiceException, rospy.ROSException) as e:
            res.success = False
            print("Service call failed: %s"%e)

        if(joint_info.position[6] <= 1.39 and joint_info.position[6] >= -0.42):
            res.success = True
            rospy.loginfo("grasped object")
        else:
            res.success = False
            rospy.loginfo("no object grasped")

        return res

    def vx300s_open(self, req):

        res = TriggerResponse()

        try:
            self.gripper.open(2.0)
            res.success = True
        except (rospy.ServiceException, rospy.ROSException) as e:
            res.success = False
            print("Service call failed: %s"%e)
        
        return res

    def vx300s_close(self, req):

        res = TriggerResponse()

        try:
            self.gripper.close(2.0)
            res.success = True
        except (rospy.ServiceException, rospy.ROSException) as e:
            res.success = False
            print("Service call failed: %s"%e)
        
        return res

    def get_pose(self, req):

        res = cur_poseResponse()

        try:
            trans, rot = self.listener.lookupTransform("vx300s/base_link", "vx300s/ee_gripper_link", rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            print("Service call failed: %s"%e)

        res.pose.position.x = trans[0]
        res.pose.position.y = trans[1]
        res.pose.position.z = trans[2]
        res.pose.orientation.x = rot[0]
        res.pose.orientation.y = rot[1]
        res.pose.orientation.z = rot[2]
        res.pose.orientation.w = rot[3]

        return res

    def vx300s_ee_pose(self, req):

        res = ee_poseResponse()

        try:
            x = req.target_pose.position.x
            y = req.target_pose.position.y
            z = req.target_pose.position.z
            ox = req.target_pose.orientation.x
            oy = req.target_pose.orientation.y
            oz = req.target_pose.orientation.z
            ow = req.target_pose.orientation.w
            roll, pitch, yaw = tfm.euler_from_quaternion([ox, oy, oz, ow])
            self.arm.set_ee_pose_components(x=x, y=y, z=z, roll=roll, pitch=pitch, yaw=yaw)
            res.result = "success"
        except (rospy.ServiceException, rospy.ROSException) as e:
            res.result = "Fail"
            print("Service call failed: %s"%e)
        
        return res

    def rot(self, req):

        res = TriggerResponse()

        try:
            # self.arm.set_ee_cartesian_trajectory(yaw=0.523)
            rot_ee_pose = [0.54, 0.00, 0.4, 0, 0, 0]
            for i in range(1,6):
                self.arm.set_ee_pose_components(x=rot_ee_pose[0], y=rot_ee_pose[1], z=rot_ee_pose[2], roll=rot_ee_pose[3], pitch=rot_ee_pose[4], yaw=0.098125*i)
            res.success = True
        except (rospy.ServiceException, rospy.ROSException) as e:
            res.success = False
            print("Service call failed: %s"%e)

        return res

    def mm_home(self, req):

        res = TriggerResponse()

        try:
            joint_positions = [-0.03681553900241852, -1.1627575159072876, 0.4832039475440979, 0.0015339808305725455, 0.6841554641723633, 0.023009711876511574]
            self.arm.set_joint_positions(joint_positions)
            res.success = True
        except (rospy.ServiceException, rospy.ROSException) as e:
            res.success = False
            print("Service call failed: %s"%e)
        
        return res

if __name__=='__main__':

    rospy.init_node("vx300s_control_node", anonymous=False)

    robot_name = rospy.get_param("robot_name")
    VX300s = vx300s(robot_name)
    
    rospy.spin()