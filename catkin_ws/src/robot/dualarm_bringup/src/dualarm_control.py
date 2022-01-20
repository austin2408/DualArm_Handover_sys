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
        rospy.Service("/{0}/go_rise".format(name), Trigger, self.vx300s_rise)
        rospy.Service("/{0}/go_handover".format(name), Trigger, self.vx300s_handover)
        rospy.Service("/{0}/go_place".format(name), Trigger, self.vx300s_place)

        # vx300s setup
        robot = InterbotixManipulatorXS(robot_model="vx300s", group_name="arm", gripper_name="gripper", robot_name=name, init_node=False)

        self.arm = robot.arm
        self.gripper = robot.gripper

        if self.name == 'right_arm':
            self.rise_pose = [-0.04908738657832146, -0.5660389065742493, 0.5460971593856812, 0.05522330850362778, -0.21629129350185394, -0.012271846644580364]
            self.handover_pose = [0.7409127354621887, -0.5476311445236206, 1.5309128761291504, -0.34514567255973816, -0.9418642520904541, 0.1288543939590454]
            self.place_pose = [-0.03834952041506767, 0.39730104804039, 1.260932207107544, 0.010737866163253784, -1.624485731124878, 0.02454369328916073]
        else:
            self.rise_pose = [-0.029145635664463043, -0.1871456652879715, 0.11351457983255386, -0.13345633447170258, -0.07516506314277649, 0.12271846830844879]
            self.handover_pose = [-0.5138835906982422, -0.41264083981513977, 1.1259418725967407, 0.052155349403619766, -0.5276893973350525, 0.0015339808305725455]
            self.place_pose = [0.6028544902801514, 1.0400390625, -0.7010292410850525, 1.1566215753555298, -0.6350680589675903, -1.055378794670105]


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

    def vx300s_rise(self, req):
        res = TriggerResponse()

        self.arm.set_joint_positions(self.rise_pose)
        res.success = True
        
        return res

    def vx300s_place(self, req):
        res = TriggerResponse()

        self.arm.set_joint_positions(self.place_pose)
        res.success = True
        
        return res

    def vx300s_handover(self, req):
        res = TriggerResponse()

        self.arm.set_joint_positions(self.handover_pose)
        res.success = True

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