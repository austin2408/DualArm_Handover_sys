import os
import numpy as np
import cv2
from scipy import ndimage
# import matplotlib.pyplot as plt
import torch
from torchvision import  transforms

from .prioritized_memory import Transition

path = os.getcwd()

image_net_mean = np.array([0.485, 0.456, 0.406])
image_net_std  = np.array([0.229, 0.224, 0.225])
transform = transforms.Compose([
                        transforms.ToTensor(),
                    ])

def hdf5_to_memory(f, memory):
    for key in f.keys():
        group = f[key]
        color = group['state/color'][()]
        depth = group['state/depth'][()]
        pixel_index = group['action'][()]
        reward = group['reward'][()]
        next_color = group['next_state/color'][()]
        next_depth = group['next_state/depth'][()]
        is_empty = group['next_state/empty'][()]

        transition = Transition(
            color, depth, pixel_index, reward, next_color, next_depth, is_empty)
        memory.add(transition)

def sample_data(memory, batch_size):
    done = False
    mini_batch = []
    idxs = []
    is_weight = []
    while not done:
        success = True
        mini_batch, idxs, is_weight = memory.sample(batch_size)
        for transition in mini_batch:
            success = success and isinstance(transition, Transition)
        if success:
            done = True
    return mini_batch, idxs, is_weight


def get_action_info(pixel_index):
    action_str = "grasp"
    rotate_idx = pixel_index[0]-2
    return action_str, rotate_idx


def standarization(prediction):
    if prediction.shape[0] != 1:
        for i in range(prediction.shape[0]):
            mean = np.nanmean(prediction[i])
            std = np.nanstd(prediction[i])
            prediction[i] = (prediction[i]-mean)/std
    else:
        mean = np.nanmean(prediction)
        std = np.nanstd(prediction)
        prediction = (prediction-mean)/std
    return prediction


def vis_affordance(predictions):
    tmp = np.copy(predictions)
    # View the value as probability
    tmp[tmp < 0] = 0
    tmp /= 5
    tmp[tmp > 1] = 1
    tmp = (tmp*255).astype(np.uint8)
    tmp.shape = (tmp.shape[0], tmp.shape[1], 1)
    heatmap = cv2.applyColorMap(tmp, cv2.COLORMAP_JET)
    return heatmap

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



def plot_figures(tool, color, depth, show=False, save=False, ros=False):
    combine = []
    tool_cmap = []
    tt = []
    i = 0
    max_ = []
    pos = []
    theta_ = [90, -45, 0, 45]
    theta_ros = [90, 45, 0, -45]
    theta_ur5 = [-0, 315, -90, -135]
    for object in tool:
        tool_cmap_ = vis_affordance(object)
        combine_ = cv2.addWeighted(color, 1.0, tool_cmap_, 0.8, 0.0)
        best = np.where(object == np.max(object))
        maxx = np.max(object)
        u, v = best[1][0], best[0][0]
        pos.append([u, v])
        combine_ = cv2.circle(combine_, (u, v), 3, (0, 0, 0), 2)
        tt_ = color.copy()
        tool_cmap.append(tool_cmap_)
        combine.append(combine_)
        tt.append(tt_)
        # print('angle : ', theta_[i], ' ',(u, v), ' max : ',maxx)
        max_.append(maxx)
        i += 1

    Max = max(max_)
    # if ros:
    #     angle = theta_ros[max_.index(Max)]
    # else:
    #     angle = theta_[max_.index(Max)]
    
    angle = theta_ros[max_.index(Max)]
    positions = pos[max_.index(Max)]
    # if show:
    #     f, axarr = plt.subplots(2, 5, figsize=(15, 8))
    #     plt.suptitle('Result : Angle : '+str(angle) +
    #                  ' Position : '+str(positions))
    #     axarr[0][0].set_title('color and depth')
    #     axarr[0][0].imshow(tt[0][:, :, [2, 1, 0]])
    #     axarr[1][0].imshow(depth)

    #     axarr[0][1].set_title('90')
    #     axarr[0][1].imshow(combine[0][:, :, ::-1])
    #     axarr[1][1].imshow(tool_cmap[0][:, :, [2, 1, 0]])

    #     axarr[0][2].set_title('-45')
    #     axarr[0][2].imshow(combine[1][:, :, ::-1])
    #     axarr[1][2].imshow(tool_cmap[1][:, :, [2, 1, 0]])

    #     axarr[0][3].set_title('0')
    #     axarr[0][3].imshow(combine[2][:, :, ::-1])
    #     axarr[1][3].imshow(tool_cmap[2][:, :, [2, 1, 0]])

    #     axarr[0][4].set_title('45')
    #     axarr[0][4].imshow(combine[3][:, :, ::-1])
    #     axarr[1][4].imshow(tool_cmap[3][:, :, [2, 1, 0]])

    #     plt.show()
    # if save:
    #     plt.savefig(path+'/result/sample.png', dpi=300)

    return [angle, positions[1], positions[0]], Max, combine[max_.index(Max)]


def preprocessing(color, depth):

    # Zoom 2 times
    color_img_2x = ndimage.zoom(color, zoom=[2, 2, 1], order=0)
    depth_img_2x = ndimage.zoom(depth, zoom=[2, 2],    order=0)

    # Add extra padding to handle rotations inside network
    diag_length = float(color_img_2x.shape[0])*np.sqrt(2)
    diag_length = np.ceil(diag_length/32)*32  # Shrink 32 times in network
    padding_width = int((diag_length - color_img_2x.shape[0])/2)

    # Convert BGR (cv) to RGB
    color_img_2x_b = np.pad(
        color_img_2x[:, :, 0], padding_width, 'constant', constant_values=0)
    color_img_2x_b.shape = (
        color_img_2x_b.shape[0], color_img_2x_b.shape[1], 1)
    color_img_2x_g = np.pad(
        color_img_2x[:, :, 1], padding_width, 'constant', constant_values=0)
    color_img_2x_g.shape = (
        color_img_2x_g.shape[0], color_img_2x_g.shape[1], 1)
    color_img_2x_r = np.pad(
        color_img_2x[:, :, 2], padding_width, 'constant', constant_values=0)
    color_img_2x_r.shape = (
        color_img_2x_r.shape[0], color_img_2x_r.shape[1], 1)
    color_img_2x = np.concatenate(
        (color_img_2x_r, color_img_2x_g, color_img_2x_b), axis=2)
    depth_img_2x = np.pad(depth_img_2x, padding_width,
                          'constant', constant_values=0)

    # Normalize color image with ImageNet data
    image_mean = [0.33638567, 0.33638567, 0.33638567]
    image_std = [0.2603763,  0.2443466,  0.24258484]
    # image_mean = [0.485, 0.456, 0.406] # for sim: [0.20414721, 0.17816422, 0.15419899]
    # image_std  = [0.229, 0.224, 0.225] # for sim: [0.1830081 , 0.16705943, 0.17520182]
    input_color_img = color_img_2x.astype(float)/255  # np.uint8 to float
    for c in range(3):
        input_color_img[:, :, c] = (
            input_color_img[:, :, c] - image_mean[c]) / image_std[c]

    # depth to meter
    tmp = depth_img_2x.astype(float)/1000.0  # to meter
    # depth thres hold
    tmp[tmp > 2] = 2

    # Normalize depth image
    # depth_mean = 1.3136337  # for sim: 0.032723393
    # depth_std = 1.9633287  # for sim: 0.056900032
    # tmp = (tmp-depth_mean)/depth_std

    # Duplicate channel to DDD
    tmp.shape = (tmp.shape[0], tmp.shape[1], 1)
    input_depth_img = np.concatenate((tmp, tmp, tmp), axis=2)

    # Convert to tensor
    # H, W, C - > N, C, H, W
    input_color_img.shape = (
        input_color_img.shape[0], input_color_img.shape[1], input_color_img.shape[2], 1)
    input_depth_img.shape = (
        input_depth_img.shape[0], input_depth_img.shape[1], input_depth_img.shape[2], 1)
    input_color_data = torch.from_numpy(
        input_color_img.astype(np.float32)).permute(3, 2, 0, 1)
    input_depth_data = torch.from_numpy(
        input_depth_img.astype(np.float32)).permute(3, 2, 0, 1)

    return input_color_data, input_depth_data, padding_width

def postProcessing(prediction, color, depth, color_tensor, pad, show=False):
    size = color.shape[0]
    s = color_tensor.shape[2]
    tool_0 = prediction[0][0, 0, pad//2:size+pad//2, pad//2:size+pad//2].detach().cpu().numpy() 
    tool_1 = prediction[1][0, 0, pad//2:size+pad//2, pad//2:size+pad//2].detach().cpu().numpy() 
    tool_2 = prediction[2][0, 0, pad//2:size+pad//2, pad//2:size+pad//2].detach().cpu().numpy() 
    tool_3 = prediction[3][0, 0, pad//2:size+pad//2, pad//2:size+pad//2].detach().cpu().numpy()

    result, value, affordance = plot_figures([tool_0, tool_1, tool_2, tool_3], color, depth, show=show, ros=True)
    
    return result, value, affordance
