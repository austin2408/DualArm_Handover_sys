import os
from model import HANet
import numpy as np
import cv2
from scipy import ndimage
import torch
import rospkg
from torchvision import  transforms
from cv_bridge import CvBridge, CvBridgeError
from scipy.spatial.transform import Rotation
from tf import TransformListener, TransformerROS, transformations
import tf
from vx300s_bringup.srv import *

# from .prioritized_memory import Transition

path = os.getcwd()

image_net_mean = np.array([0.485, 0.456, 0.406])
image_net_std  = np.array([0.229, 0.224, 0.225])
transform = transforms.Compose([
                        transforms.ToTensor(),
                    ])

def processing(color, depth):
    color = cv2.resize(color, (224, 224))
    depth = cv2.resize(depth, (224, 224))
    depth[depth > 1000] = 0.0

    color= color[:,:,[2,1,0]]
    color = (color/255.).astype(float)
    color_rgb = np.zeros(color.shape)
    # depth[depth > 1000] = 0

    for i in range(3):
        color_rgb[:, :, i] = (color[:, :, 2-i]-image_net_mean[i])/image_net_std[i]

    depth = np.round((depth/np.max(depth))*255).astype('int').reshape(1,depth.shape[0],depth.shape[1])
    # depth[depth > 1000] = 0

    depth = (depth/1000.).astype(float) # to meters
    depth = np.clip(depth, 0.0, 1.2)
    depth_3c = np.zeros(color.shape)

    for i in range(3):
        depth_3c[:, :, i] = (depth[:, :]-image_net_mean[i])/image_net_std[i]

    c = transform(color_rgb)
    d = transform(depth_3c)
    c = torch.unsqueeze(c,0)
    d = torch.unsqueeze(d,0)
    # c = torch.float(c)
    # d = torch.float(d)

    return c.float(), d.float()


def aff_process(pred, color, depth):
    graspable = cv2.resize(pred, (640, 480))
    graspable[depth==0] = 0
    graspable[graspable>=1] = 0.99999
    graspable[graspable<0] = 0
    graspable = cv2.GaussianBlur(graspable, (7, 7), 0)
    affordanceMap = (graspable/np.max(graspable)*255).astype(np.uint8)
    affordanceMap = cv2.applyColorMap(affordanceMap, cv2.COLORMAP_JET)
    affordanceMap = affordanceMap[:,:,[2,1,0]]
    combine = cv2.addWeighted(color,0.7,affordanceMap, 0.3,0)

    gray = cv2.cvtColor(affordanceMap, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    binaryIMG = cv2.Canny(blurred, 20, 160)
    contours, _ = cv2.findContours(binaryIMG, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    i = 0
    point_x = 0
    point_y = 0
    cX = 0
    cY = 0
    x = 0
    y = 0
    
    for c in contours:
        M = cv2.moments(c)
        if(M["m00"]!=0): 
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            zc = depth[cY, cX]/1000
            if 0 < zc < 0.65:
                i += 1
                point_x += cX
                point_y += cY

    if i != 0:
        x = int(point_x / i)
        y = int(point_y / i)

    return x, y, combine

class Affordance_predict():
    def __init__(self, arm, fx, fy, cx, cy):
        r = rospkg.RosPack()
        self.path = r.get_path("handover")
        self.net = HANet(4)
        self.net.load_state_dict(torch.load(self.path+'/src/ddqn/weight/HANet.pth'))
        self.net = self.net.cuda()
        self.bridge = CvBridge()
        self.target_cam_dis = 1000
        self.listener = TransformListener()
        self.transformer = TransformerROS()
        self.arm = arm
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        

    def predict(self, color, depth):
        # Convert msg type
        A = [90,45,0,-45]
        
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
            # p = self.bridge.cv2_to_imgmsg(aff_pub, "bgr8")
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
            
            return target_pose, go_ok, self.target_cam_dis
        else:
            return None, False, self.target_cam_dis

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

        return tf_pose, vaild

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

