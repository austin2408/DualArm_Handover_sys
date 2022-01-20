#!/usr/bin/env python3

from ddqn.options import Option
from ddqn.agent import Agent
from ddqn.prioritized_memory import Transition
import rospkg
import message_filters
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import CameraInfo, CompressedImage, Image
from geometry_msgs.msg import PoseStamped
from std_srvs.srv import Trigger, TriggerResponse, TriggerRequest
import rospy
import os
import cv2
import copy
import numpy as np
from scipy.spatial.transform import Rotation
import warnings
from tf import TransformListener, TransformerROS, transformations
import tf
from handover.srv import *
warnings.filterwarnings("ignore")

empty_buffer = {
    'color': None, 'depth': None, 'pixel_idx': None, 'reward': None, 'next_color': None, 'next_depth': None, 'is_empty': None
}

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
        self.trans_buf = copy.deepcopy(empty_buffer)
        self.face_buf = []
        self.train = False
        self.arm = 'left_arm'
        self.listener = TransformListener()
        self.transformer = TransformerROS()
        info = rospy.wait_for_message('camera_right/color/camera_info', CameraInfo)
        self.fx = info.P[0]
        self.fy = info.P[5]
        self.cx = info.P[2]
        self.cy = info.P[6]


        # ddqn agent
        agentArgs = Option().create(os.path.join(self.path, 'src/ddqn/config/continuous.yml'))
        self.agent = Agent(agentArgs)
        self.agent.load_pretrained(os.path.join(self.path, 'src/ddqn/weight/behavior_500_0.06.pth'))

        # Publisher
        self.pred_img_pub = rospy.Publisher('~prediction', Image, queue_size=1)

        # Mssage filter
        self.color_right = message_filters.Subscriber('/camera_right/color/image_raw/compressed', CompressedImage)
        self.depth_right = message_filters.Subscriber('/camera_right/aligned_depth_to_color/image_raw', Image)

        self.color_left = message_filters.Subscriber('/camera_left/color/image_raw/compressed', CompressedImage)
        self.depth_left = message_filters.Subscriber('/camera_left/aligned_depth_to_color/image_raw', Image)

        ts = message_filters.ApproximateTimeSynchronizer([self.color_right, self.depth_right, self.color_left, self.depth_left], 5, 5)
        ts.registerCallback(self.callback_msgs)

        # Service
        rospy.Service('~grasp', Trigger, self.strategy)
        # update_server = rospy.Service('~update_server', SetBool, self.update)
        self.reset_arm()

    def reset_arm(self):
        r = TriggerRequest()

        try:
            go_initial = rospy.ServiceProxy("/{0}/go_rise".format('right_arm'), Trigger)
            resp = go_initial(r)
            # go_pose = rospy.ServiceProxy("/{0}/gripper_open".format('right_arm'), Trigger)
            # resp = go_pose(r)

            go_initial = rospy.ServiceProxy("/{0}/go_rise".format('left_arm'), Trigger)
            resp = go_initial(r)
            # go_pose = rospy.ServiceProxy("/{0}/gripper_open".format('left_arm'), Trigger)
            # resp = go_pose(r)

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
            rospy.loginfo('Giver : left_arm / Taker : right_arm')
            target, GO = self.predict(self.color_right, self.depth_right)
        else:
            rospy.loginfo('Giver : right_arm / Taker : left_arm')
            target, GO = self.predict(self.color_left, self.depth_left)

        print(target)

        self.check_depth()


        # Open gripper
        # try:
        #     go_pose = rospy.ServiceProxy("/{0}/gripper_open".format(self.arm), Trigger)
        #     resp = go_pose(r)
        #     res.success = True
        # except rospy.ServiceException as exc:
        #     res.success = False
        #     print("service did not process request: " + str(exc))

        # Grasp
        # if GO:
        #     try:
        #         go_pose = rospy.ServiceProxy("/{0}/go_pose".format(self.arm), ee_pose)
        #         resp = go_pose(target)
        #         res.success = True
        #     except rospy.ServiceException as exc:
        #         res.success = False
        #         print("service did not process request: " + str(exc))

        # rospy.sleep(1)

        # Close gripper
        # try:
        #     go_pose = rospy.ServiceProxy("/{0}/gripper_close".format(self.arm), Trigger)
        #     resp = go_pose(r)
        #     res.success = True
        # except rospy.ServiceException as exc:
        #     res.success = False
        #     print("service did not process request: " + str(exc))

        
        # if self.arm == 'right_arm':
        #     try:
        #         go_pose = rospy.ServiceProxy("/{0}/gripper_open".format('left_arm'), Trigger)
        #         resp = go_pose(r)
        #         res.success = True
        #     except rospy.ServiceException as exc:
        #         res.success = False
        #         print("service did not process request: " + str(exc))
        # else:
        #     try:
        #         go_pose = rospy.ServiceProxy("/{0}/gripper_open".format('right_arm'), Trigger)
        #         resp = go_pose(r)
        #         res.success = True
        #     except rospy.ServiceException as exc:
        #         res.success = False
        #         print("service did not process request: " + str(exc))

        # Reset
        # try:
        #     go_initial = rospy.ServiceProxy("/{0}/go_rise".format(self.arm), Trigger)
        #     resp = go_initial(r)
        #     res.success = True
        # except rospy.ServiceException as exc:
        #     res.success = False
        #     print("service did not process request: " + str(exc))

        # if self.arm == 'right_arm':
        #     self.arm = 'left_arm'
        # else:
        #     self.arm = 'right_arm'

        return res



    def predict(self, color, depth):
        # Convert msg type
        try:
            cv_image = self.bridge.compressed_imgmsg_to_cv2(
                color, "bgr8")
            cv_depth = self.bridge.imgmsg_to_cv2(depth, "16UC1")
            cv_depth_grasp = cv_depth.copy()

        except CvBridgeError as e:
            print(e)
            return

        # Do prediction
        image_pub = cv_image.copy()

        cv_image = cv2.resize(cv_image, (224, 224))
        cv_depth = cv2.resize(cv_depth, (224, 224))
        action, value, _ = self.agent.inference(cv_image, cv_depth)


        # Get gripping point base on camera link
        x = action[2]/224.0*640
        y = action[1]/224.0*480
        z = cv_depth_grasp[int(y), int(x)]/1000.0

        image_pub = cv2.circle(image_pub, (int(x), int(y)), 5, (0,255,0), 0)

        p = self.bridge.cv2_to_imgmsg(image_pub, "bgr8")

        self.pred_img_pub.publish(p)

        camera_x, camera_y, camera_z = self.getXYZ(x, y, z)

        rot = Rotation.from_euler('xyz', [0, 0, 0], degrees=True)
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

        target_pose, go_ok = self.camera2world(Target_pose, self.arm)

        if self.train:
            action[0] = int(abs(action[0]-90)/45)
            self.face_buf.append({'color': copy.deepcopy(cv_image), 'depth': copy.deepcopy(
            cv_depth), 'pixel_idx': copy.deepcopy(action), 'value': value})
            indx = max(range(len(self.face_buf)),key=lambda index: self.face_buf[index]['value'])

            self.trans_buf['color'] = self.face_buf[indx]['color']
            self.trans_buf['depth'] = self.face_buf[indx]['depth']
            self.trans_buf['pixel_idx'] = self.face_buf[indx]['pixel_idx']

        return target_pose, go_ok


    def camera2world(self, camera_pose, arm):
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
            if self.arm == 'left_arm':
                tf_pose.target_pose.position.x = tf_pose_matrix[0, 3] - 0.04
                tf_pose.target_pose.position.y = tf_pose_matrix[1, 3] + 0.045
                tf_pose.target_pose.position.z = tf_pose_matrix[2, 3] + 0.035
            else:
                tf_pose.target_pose.position.x = tf_pose_matrix[0, 3] - 0.3
                tf_pose.target_pose.position.y = tf_pose_matrix[1, 3] + 0.03
                tf_pose.target_pose.position.z = tf_pose_matrix[2, 3] + 0.035

            tf_pose.target_pose.orientation.x = rot_quat[0]
            tf_pose.target_pose.orientation.y = rot_quat[1]
            tf_pose.target_pose.orientation.z = rot_quat[2]
            tf_pose.target_pose.orientation.w = rot_quat[3]

        if tf_pose.target_pose.position.x > 0.7:
            vaild = False

        return tf_pose, vaild

    def check_depth(self):
        if self.arm == 'left_arm':
            Depth = self.bridge.imgmsg_to_cv2(self.depth_right, "16UC1")
        else:
            Depth = self.bridge.imgmsg_to_cv2(self.depth_left, "16UC1")

        print(Depth)
        find = np.where(Depth==0)
        print(find[0].shape)
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

            # res.success = True
        except rospy.ServiceException as exc:
            # res.success = False
            print("service did not process request: " + str(exc))

        rospy.loginfo("Shutdown.")



if __name__ == '__main__':
    rospy.init_node("dualarm_handover", anonymous=False)
    dualarm_RL = DualArm_Handover()
    rospy.on_shutdown(dualarm_RL.onShutdown)
    rospy.spin()
