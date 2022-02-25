#!/usr/bin/env python3

from HANet.model import HANet
from HANet.utils import processing, aff_process
import rospkg
import message_filters
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import CameraInfo, CompressedImage, Image
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import PoseStamped, WrenchStamped
from std_srvs.srv import Trigger, TriggerResponse, TriggerRequest
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
        self.f_x = 0
        self.f_y = 0
        self.f_z = 0
        self.target_cam_dis = 1000
        self.dis_decay = 0.5


        # Load model
        self.net = HANet(4)
        self.net.load_state_dict(torch.load(self.path+'/src/ddqn/weight/HANet.pth'))
        self.net = self.net.cuda()

        # Publisher
        self.pred_img_pub = rospy.Publisher('~prediction/image', Image, queue_size=1)
        self.dis_pub_right = rospy.Publisher('~right_arm/target_cam_dis', Float32MultiArray, queue_size=1)
        self.dis_pub_left = rospy.Publisher('~left_arm/target_cam_dis', Float32MultiArray, queue_size=1)

        # Mssage filter
        self.color_right = message_filters.Subscriber('/camera_right/color/image_raw/compressed', CompressedImage)
        self.depth_right = message_filters.Subscriber('/camera_right/aligned_depth_to_color/image_raw', Image)

        self.color_left = message_filters.Subscriber('/camera_left/color/image_raw/compressed', CompressedImage)
        self.depth_left = message_filters.Subscriber('/camera_left/aligned_depth_to_color/image_raw', Image)

        ts = message_filters.ApproximateTimeSynchronizer([self.color_right, self.depth_right, self.color_left, self.depth_left], 5, 5)
        ts.registerCallback(self.callback_msgs)

        # Subscrib Force Sensor
        self.force = rospy.Subscriber("/robotiq_ft_wrench", WrenchStamped, self.force_detect)

        # Service
        rospy.Service('~grasp_pa', Trigger, self.Static_Passive_strategy)
        rospy.Service('~cl_grasp_pa', Trigger, self.Dynamic_Passive_strategy)
        rospy.Service('~grasp_in', Trigger, self.Static_Active_strategy)
        rospy.Service('~cl_grasp_in', Trigger, self.Dynamic_Active_strategy)
        rospy.Service('~change_hand', Trigger, self.switch_srv)

        # Initial
        self.reset_arm()
        rospy.loginfo('System startup is complete ! The Taker is '+self.arm)

    # Switch opterated arm
    def switch(self):
        if self.arm == 'right_arm':
            self.arm = 'left_arm'
        else:
            self.arm = 'right_arm'

        rospy.loginfo('============= The Taker is switched to '+self.arm+ '=============')

    def force_detect(self, msg):
        self.f_x = int(msg.wrench.force.x)
        self.f_y = int(msg.wrench.force.y)
        self.f_z = int(msg.wrench.force.z)


    def switch_srv(self, req):
        res = TriggerResponse()

        self.switch()

        res.success = True

        return res

    def pub_distance(self, sign):
        Dis = Float32MultiArray()
        Dis.data = [sign, self.target_cam_dis]

        if self.arm == 'right_arm':
            self.dis_pub_right.publish(Dis)
        else:
            self.dis_pub_left.publish(Dis)

    def check_gripper(self):
        x = self.f_x

        while True:
            if x != self.f_x:
                self.close_gripper(self.arm)
                break


    def reset_arm(self):
        self.handover_init('right_arm')
        self.open_gripper('right_arm')

        self.handover_init('left_arm')
        self.open_gripper('left_arm')

        rospy.loginfo('Reset arms to initial state')


    def callback_msgs(self, colorR, depthR, colorL, depthL):
        self.color_right = colorR
        self.depth_right = depthR

        self.color_left = colorL
        self.depth_left = depthL

    # Deal with Static / Passive scenario
    def Static_Passive_strategy(self, req):
        res = TriggerResponse()

        r = TriggerRequest()
        
        # Make prediction
        rospy.loginfo('============================================')

        target, GO = self.predict()

        if target == None:
            res.success = False
            rospy.loginfo('Grasping Failed ! ')

            return res

        print(target)

        # Open gripper
        self.open_gripper(self.arm)

        # Grasp
        if GO:
            try:
                go_pose = rospy.ServiceProxy("/{0}/go_pose".format(self.arm), ee_pose)
                resp = go_pose(target)
                res.success = True
            except rospy.ServiceException as exc:
                res.success = False
                print("service did not process request: " + str(exc))

            rospy.sleep(1)

            self.close_gripper(self.arm)

            rospy.sleep(0.5)

            self.place(self.arm)

            self.open_gripper(self.arm)

            self.mid(self.arm)

            self.handover_init(self.arm)

            rospy.loginfo('Grasping Complete')

        return res

    # Deal with Static / Active scenario
    def Static_Active_strategy(self, req):
        self.arm = 'right_arm'

        res = TriggerResponse()

        r = TriggerRequest()

        self.go_loop = True
        self.dis_decay = 0.6
        
        # Make prediction
        rospy.loginfo('============================================')

        target, GO = self.predict()

        if target == None:
            res.success = False
            rospy.loginfo('Grasping Failed ! ')

            return res

        print(target)

        # Open gripper
        self.open_gripper(self.arm)

        # Grasp
        if GO:
            try:
                go_pose = rospy.ServiceProxy("/{0}/go_pose".format(self.arm), ee_pose)
                resp = go_pose(target)
                res.success = True
            except rospy.ServiceException as exc:
                res.success = False
                print("service did not process request: " + str(exc))

            rospy.sleep(1.5)

            print("Waiting ......")
            self.check_gripper()

            rospy.sleep(0.5)

            self.place(self.arm)

            self.open_gripper(self.arm)

            self.mid(self.arm)

            self.handover_init(self.arm)

            rospy.loginfo('Grasping Complete')

        self.dis_decay = 1.0
        self.go_loop = False

        return res

    # Deal with Dynamic / Passive scenario
    def Dynamic_Passive_strategy(self, req):
        res = TriggerResponse()

        self.open_gripper(self.arm)

        test_count = 0

        self.go_loop = True

        while self.go_loop:
            rospy.loginfo('Loop '+str(test_count+1))
            print(self.target_cam_dis)

            target, GO = self.predict()

            if GO and target!=None:
                try:
                    go_pose = rospy.ServiceProxy("/{0}/go_pose".format(self.arm), ee_pose)
                    print(target)
                    resp = go_pose(target)
                    test_count += 1
                    if self.target_cam_dis < 0.04:
                        self.pub_distance(1)
                        self.go_loop = False
                    else:
                        self.pub_distance(0)
                    # break
                except rospy.ServiceException as exc:
                    print("service did not process request: " + str(exc))


            rospy.sleep(1)

        self.close_gripper(self.arm)

        rospy.sleep(0.5)

        self.place(self.arm)

        self.open_gripper(self.arm)

        self.mid(self.arm)

        self.handover_init(self.arm)

        rospy.loginfo('Grasping Complete')

        return res

    # Deal with Dynamic / Active scenario
    def Dynamic_Active_strategy(self, req):
        self.arm = 'right_arm'
        res = TriggerResponse()


        self.open_gripper(self.arm)

        test_count = 0

        self.go_loop = True

        while self.go_loop:
            rospy.loginfo('Loop '+str(test_count+1))
            print(self.target_cam_dis)

            target, GO = self.predict()

            if GO and target!=None:
                try:
                    go_pose = rospy.ServiceProxy("/{0}/go_pose".format(self.arm), ee_pose)
                    resp = go_pose(target)
                    test_count += 1
                    if self.target_cam_dis < 0.08:
                        self.pub_distance(1)
                        self.go_loop = False
                    else:
                        self.pub_distance(0)
                except rospy.ServiceException as exc:
                    print("service did not process request: " + str(exc))

            else:        
                test_count += 1

            rospy.sleep(1)

        rospy.sleep(1.5)
        print("Waiting ......")
        self.check_gripper()

        rospy.sleep(0.5)

        self.place(self.arm)

        self.open_gripper(self.arm)

        self.mid(self.arm)

        self.handover_init(self.arm)

        rospy.loginfo('Grasping Complete')

        return res


    def close_gripper(self, arm: str):
        r = TriggerRequest()

        try:
            go_pose = rospy.ServiceProxy("/{0}/gripper_close".format(arm), Trigger)
            resp = go_pose(r)
        except rospy.ServiceException as exc:
            print("service did not process request: " + str(exc))

    def place(self, arm: str):
        r = TriggerRequest()

        try:
            go_pose = rospy.ServiceProxy("/{0}/go_place".format(arm), Trigger)
            resp = go_pose(r)
        except rospy.ServiceException as exc:
            print("service did not process request: " + str(exc))

    # Move to relay position
    def mid(self, arm: str):
        r = TriggerRequest()

        try:
            go_pose = rospy.ServiceProxy("/{0}/go_sleep".format(arm), Trigger)
            resp = go_pose(r)
        except rospy.ServiceException as exc:
            print("service did not process request: " + str(exc))

    def handover_init(self, arm: str):
        r = TriggerRequest()

        try:
            go_pose = rospy.ServiceProxy("/{0}/go_handover".format(arm), Trigger)
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
        A = [90,45,0,-45]

        if self.arm == 'right_arm':
            color = self.color_right
            depth = self.depth_right
        else:
            color = self.color_left
            depth = self.depth_left
        
        # Convert msg type
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
            self.pred_img_pub.publish(p)

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

        

        if self.go_loop:
            try:
                if self.arm == 'right_arm':
                    self.listener.waitForTransform('right_arm/base_link', 'right_arm/ee_arm_link', rospy.Time(0), rospy.Duration(1.0))
                    (trans, rot) = self.listener.lookupTransform('right_arm/base_link', 'right_arm/ee_arm_link', rospy.Time(0))
                else:
                    self.listener.waitForTransform('left_arm/base_link', 'left_arm/ee_arm_link', rospy.Time(0), rospy.Duration(1.0))
                    (trans, rot) = self.listener.lookupTransform('left_arm/base_link', 'left_arm/ee_arm_link', rospy.Time(0))

            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                print("Error TF listening")
                return
            tf_pose.target_pose.position.x = (tf_pose.target_pose.position.x + trans[0])*self.dis_decay
            tf_pose.target_pose.position.y = (tf_pose.target_pose.position.y + trans[1])*self.dis_decay
            tf_pose.target_pose.position.z = (tf_pose.target_pose.position.z + trans[2])*self.dis_decay


        if tf_pose.target_pose.position.x < 0.0:
            tf_pose.target_pose.position.x = 0.1

        return tf_pose, vaild

    def check_depth(self):
        if self.arm == 'left_arm':
            Depth = self.bridge.imgmsg_to_cv2(self.depth_right, "16UC1")
        else:
            Depth = self.bridge.imgmsg_to_cv2(self.depth_left, "16UC1")

        # print(Depth)
        find = np.where(Depth==0)
        print(find[0].shape)
        # print("Min of depth : ", np.min(Depth), np.max(Depth))



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
