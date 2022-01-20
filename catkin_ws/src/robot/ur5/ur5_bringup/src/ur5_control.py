#!/usr/bin/env python3

import rospy
from arm_operation.srv import *
from arm_operation.msg import *
from std_srvs.srv import Trigger, TriggerResponse
from ur5_bringup.srv import *
import tf
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import math

class UR5():
    def __init__(self):
        rospy.Service("/ur5/go_home", Trigger, self.ur5_home)
        rospy.Service("/ur5/get_pose", cur_pose, self.get_pose)
        rospy.Service("/ur5/rotate", rotate, self.rotate_object)
        self.goto_pose = rospy.ServiceProxy('/ur5_control_server/ur_control/goto_pose', arm_operation.srv.target_pose) 
        self.mani_joint_srv = '/ur5_control_server/ur_control/goto_joint_pose'
        self.mani_move_srv = rospy.ServiceProxy(self.mani_joint_srv, joint_pose)
        self.mani_req = joint_poseRequest()
        self.p = joint_value()

        self.listener = tf.TransformListener()

    def rotate_object(self, req):

        res = rotateResponse()

        rad = math.pi * req.angle / 180 
        di = req.direction

        try:
            trans, _ = self.listener.lookupTransform("base_link", "object_link", rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            print("Service call failed: %s"%e)

        for i in range(1, 6):

            try:
                _, rot = self.listener.lookupTransform("base_link", "object_link", rospy.Time(0))
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
                print("Service call failed: %s"%e)

            (row, pitch, yaw) = euler_from_quaternion(rot)

            if di == "right":
                yaw += math.pi / 180
            elif di == "left":
                yaw -= math.pi / 180

            quat = quaternion_from_euler(row, pitch, yaw)

            pose = arm_operation.srv.target_poseRequest()

            pose.target_pose.position.x = trans[0] 
            pose.target_pose.position.y = trans[1]
            pose.target_pose.position.z = trans[2]
            pose.target_pose.orientation.x = quat[0]
            pose.target_pose.orientation.y = quat[1]
            pose.target_pose.orientation.z = quat[2]
            pose.target_pose.orientation.w = quat[3]

            self.goto_pose(pose)

        res.result = "Finish rotation" 
            
        return res

    def get_pose(self, req):

        res = cur_poseResponse()

        try:
            trans, rot = self.listener.lookupTransform("base_link", "object_link", rospy.Time(0))
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

    def ur5_home(self, req):
        self.p.joint_value = [0.0011875617783516645, -2.1486170927630823, 2.3022329807281494, -3.3030384222613733, -1.5724604765521448, -1.5706184546100062]
        self.mani_req.joints.append(self.p)
        res = TriggerResponse()

        try:
            rospy.wait_for_service(self.mani_joint_srv)
            mani_resp = self.mani_move_srv(self.mani_req)
            res.success = True
        except (rospy.ServiceException, rospy.ROSException) as e:
            res.success = False
            print("Service call failed: %s"%e)
        
        return res

if __name__ == "__main__":
    rospy.init_node('ur5_control_node')
    ur5 = UR5()
    rospy.spin()