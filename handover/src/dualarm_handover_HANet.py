#!/usr/bin/env python3

from HANet.model import HANet
from HANet.utils import processing, aff_process
import rospkg
import message_filters
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import CameraInfo, CompressedImage, Image
from geometry_msgs.msg import PoseStamped
from std_srvs.srv import Trigger, TriggerResponse, TriggerRequest, SetBool, SetBoolResponse
import rospy
import cv2
import torch
import numpy as np
from scipy.spatial.transform import Rotation
import warnings
from tf import TransformListener, TransformerROS, transformations
import tf
from handover.srv import *
warnings.filterwarnings("ignore")


class DualArm_Handover():
    def __init__(self):
        self.bridge = CvBridge()
        r = rospkg.RosPack()
        self.path = r.get_path("handover")

        # global variable
        self.colorMsg_right = None
        self.depthMsg_right = None
        self.colorMsg_left = None
        self.depthMsg_left = None
        self.go_loop = False
        self.arm = 'left_arm'
        self.near = 0.0
        self.listener = TransformListener()
        self.transformer = TransformerROS()
        info = rospy.wait_for_message('camera_right/color/camera_info', CameraInfo)
        self.fx = info.P[0]
        self.fy = info.P[5]
        self.cx = info.P[2]
        self.cy = info.P[6]


        # ddqn agent
        self.net = HANet(4)
        self.net.load_state_dict(torch.load(self.path+'/src/ddqn/weight/HANet.pth'))
        self.net = self.net.cuda()

        # Publisher
        self.pred_img_pub = rospy.Publisher('~prediction/image', Image, queue_size=1)

        # Mssage filter
        self.color_right = message_filters.Subscriber('/camera_right/color/image_raw/compressed', CompressedImage)
        self.depth_right = message_filters.Subscriber('/camera_right/aligned_depth_to_color/image_raw', Image)

        self.color_left = message_filters.Subscriber('/camera_left/color/image_raw/compressed', CompressedImage)
        self.depth_left = message_filters.Subscriber('/camera_left/aligned_depth_to_color/image_raw', Image)

        ts = message_filters.ApproximateTimeSynchronizer([self.color_right, self.depth_right, self.color_left, self.depth_left], 5, 5)
        ts.registerCallback(self.callback_msgs)

        # Service
        rospy.Service('~grasp', Trigger, self.strategy)
        rospy.Service('~cl_grasp', Trigger, self.Close_Loop_strategy)
        rospy.Service('~change_hand', Trigger, self.switch_srv)
        self.reset_arm()
        rospy.loginfo('System startup is complete ! The Taker is '+self.arm)

    def switch(self):
        if self.arm == 'right_arm':
            self.arm = 'left_arm'
        else:
            self.arm = 'right_arm'

        rospy.loginfo('============= The Taker is switched to '+self.arm+ '=============')


    def switch_srv(self, req):
        res = TriggerResponse()

        self.switch()

        res.success = True

        return res


    def reset_arm(self):
        r = TriggerRequest()

        try:
            go_initial = rospy.ServiceProxy("/{0}/go_handover".format('right_arm'), Trigger)
            resp = go_initial(r)
            go_pose = rospy.ServiceProxy("/{0}/gripper_open".format('right_arm'), Trigger)
            resp = go_pose(r)

            go_initial = rospy.ServiceProxy("/{0}/go_handover".format('left_arm'), Trigger)
            resp = go_initial(r)
            go_pose = rospy.ServiceProxy("/{0}/gripper_open".format('left_arm'), Trigger)
            resp = go_pose(r)

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

        grasp_flag = True

        self.near = 0.0

        # Make prediction
        rospy.loginfo('============================================')

        target, GO = self.predict()

        if target == None:
            res.success = False
            rospy.loginfo('Grasping Failed ! ')

            return res

        print(target)

        # self.check_depth()


        # Open gripper
        self.open_gripper(self.arm)

        # Grasp
        # while grasp_flag:
        if GO:
            try:
                go_pose = rospy.ServiceProxy("/{0}/go_pose".format(self.arm), ee_pose)
                resp = go_pose(target)
                res.success = True
            except rospy.ServiceException as exc:
                res.success = False
                print("service did not process request: " + str(exc))

                # grasp_flag =False

        rospy.sleep(1)

        # Close gripper
        self.close_gripper(self.arm)

        rospy.sleep(0.5)

        # Reset
        try:
            go_initial = rospy.ServiceProxy("/{0}/go_handover".format(self.arm), Trigger)
            resp = go_initial(r)
            # res.success = True
        except rospy.ServiceException as exc:
            # res.success = False
            print("service did not process request: " + str(exc))

        self.open_gripper(self.arm)

        rospy.loginfo('Grasping Complete')

        return res

    def Close_Loop_strategy(self, req):
        res = TriggerResponse()

        self.open_gripper(self.arm)

        test_count = 0

        self.go_loop = True

        self.near = -0.02
        c = 0

        while self.go_loop:
            rospy.loginfo('Loop '+str(test_count+1))

            if test_count == 2:
                self.go_loop = False

            for i in range(5):
                target, GO = self.predict()

                if GO:
                    try:
                        go_pose = rospy.ServiceProxy("/{0}/go_pose".format(self.arm), ee_pose)
                        resp = go_pose(target)
                    except rospy.ServiceException as exc:
                        print("service did not process request: " + str(exc))

                    test_count += 1
                    c = 0
                    break
                else:        
                    if i == 4:
                        test_count += 1
                
            rospy.sleep(0.5)

        self.close_gripper(self.arm)

        r = TriggerRequest()

        try:
            go_initial = rospy.ServiceProxy("/{0}/go_handover".format(self.arm), Trigger)
            resp = go_initial(r)
            res.success = True
        except rospy.ServiceException as exc:
            res.success = False
            print("service did not process request: " + str(exc))

        self.open_gripper(self.arm)

        rospy.loginfo('Grasping Complete')

        return res


    def close_gripper(self, arm: str):
        r = TriggerRequest()

        try:
            go_pose = rospy.ServiceProxy("/{0}/gripper_close".format(arm), Trigger)
            resp = go_pose(r)
        except rospy.ServiceException as exc:
            print("service did not process request: " + str(exc))

    def open_gripper(self, arm: str):
        r = TriggerRequest()
        
        try:
            go_pose = rospy.ServiceProxy("/{0}/gripper_open".format(arm), Trigger)
            resp = go_pose(r)
        except rospy.ServiceException as exc:
            print("service did not process request: " + str(exc))



    def predict(self):
        # Convert msg type
        A = [90,45,0,-45]

        if self.arm == 'right_arm':
            # rospy.loginfo('Taker : right_arm')
            color = self.color_right
            depth = self.depth_right
        else:
            # rospy.loginfo('Taker : left_arm')
            color = self.color_left
            depth = self.depth_left
        
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
            print(A[pred_id], x, y, z)

            aff_pub = cv2.circle(aff_pub, (int(x), int(y)), 10, (0,255,0), -1)
            p = self.bridge.cv2_to_imgmsg(aff_pub, "bgr8")
            self.pred_img_pub.publish(p)

            camera_x, camera_y, camera_z = self.getXYZ(x, y, z)

            if self.go_loop:
                rot = Rotation.from_euler('xyz', [0, 0, 0], degrees=True)
            else:
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
            if self.arm == 'right_arm':
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
            if self.arm == 'left_arm':
                tf_pose.target_pose.position.x = tf_pose_matrix[0, 3] + 0.02 + self.near
                tf_pose.target_pose.position.y = tf_pose_matrix[1, 3] + 0.045
                tf_pose.target_pose.position.z = tf_pose_matrix[2, 3] + 0.035
            else:
                tf_pose.target_pose.position.x = tf_pose_matrix[0, 3] - 0.01 + self.near
                tf_pose.target_pose.position.y = tf_pose_matrix[1, 3] + 0.03
                tf_pose.target_pose.position.z = tf_pose_matrix[2, 3] + 0.035

            tf_pose.target_pose.orientation.x = rot_quat[0]
            tf_pose.target_pose.orientation.y = rot_quat[1]
            tf_pose.target_pose.orientation.z = rot_quat[2]
            tf_pose.target_pose.orientation.w = rot_quat[3]

        if tf_pose.target_pose.position.x > 0.5:
            vaild = False

        if tf_pose.target_pose.position.x < 0.0:
            tf_pose.target_pose.position.x = 0.01

        return tf_pose, vaild

    def check_depth(self):
        if self.arm == 'left_arm':
            Depth = self.bridge.imgmsg_to_cv2(self.depth_right, "16UC1")
        else:
            Depth = self.bridge.imgmsg_to_cv2(self.depth_left, "16UC1")

        # print(Depth)
        find = np.where(Depth==0)
        # print(find[0].shape)
        print("Min of depth : ", np.min(Depth), np.max(Depth))



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

    def onShutdown(self):
        r = TriggerRequest()

        try:
            go_initial = rospy.ServiceProxy("/{0}/go_sleep".format('right_arm'), Trigger)
            resp = go_initial(r)

            go_initial = rospy.ServiceProxy("/{0}/go_sleep".format('left_arm'), Trigger)
            resp = go_initial(r)

        except rospy.ServiceException as exc:
            print("service did not process request: " + str(exc))

        rospy.loginfo("Shutdown.")



if __name__ == '__main__':
    rospy.init_node("dualarm_handover", anonymous=False)
    dualarm_RL = DualArm_Handover()
    rospy.on_shutdown(dualarm_RL.onShutdown)
    rospy.spin()
