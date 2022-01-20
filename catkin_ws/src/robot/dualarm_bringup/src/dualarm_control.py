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

if __name__=='__main__':

    rospy.init_node("dualarm_control_node", anonymous=False)

    robot_name = rospy.get_param("right")
    VX300s = vx300s(robot_name)

    robot_name = rospy.get_param("left")
    VX300s = vx300s(robot_name)
    
    rospy.spin()