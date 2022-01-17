import os
import numpy as np
import cv2
from scipy import ndimage
import torch
from torchvision import  transforms

from .prioritized_memory import Transition

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

