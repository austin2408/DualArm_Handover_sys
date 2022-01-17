from pickle import TRUE
import numpy as np
import copy
from numpy.lib.utils import deprecate
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
from torchvision import transforms as TF
from collections import OrderedDict
from scipy import ndimage


def rotate_heightmap(color_tensor, depth_tensor, theta, use_cuda):
    # theta in radian
    affine_mat_before = np.asarray([[np.cos(-theta), -np.sin(-theta), 0],
                                    [np.sin(-theta),  np.cos(-theta), 0]])
    affine_mat_before.shape = (2, 3, 1)
    affine_mat_before = torch.from_numpy(
        affine_mat_before).permute(2, 0, 1).float()
    if use_cuda:
        flow_grid_before = F.affine_grid(
            Variable(affine_mat_before, requires_grad=False).cuda(), color_tensor.size())
        rotate_color_tensor = F.grid_sample(
            Variable(color_tensor, volatile=True).cuda(), flow_grid_before, mode="nearest")
        rotate_depth_tensor = F.grid_sample(
            Variable(depth_tensor, volatile=True).cuda(), flow_grid_before, mode="nearest")
    else:
        flow_grid_before = F.affine_grid(
            Variable(affine_mat_before, requires_grad=False), color_tensor.size())
        rotate_color_tensor = F.grid_sample(
            Variable(color_tensor, volatile=True), flow_grid_before, mode="nearest")
        rotate_depth_tensor = F.grid_sample(
            Variable(depth_tensor, volatile=True), flow_grid_before, mode="nearest")
    return rotate_color_tensor, rotate_depth_tensor


def rotate_featuremap(feature_tensor, theta, use_cuda):
    # theta in radian
    affine_mat_after = np.asarray([[np.cos(-theta), -np.sin(-theta), 0],
                                   [np.sin(-theta),  np.cos(-theta), 0]])
    affine_mat_after.shape = (2, 3, 1)
    affine_mat_after = torch.from_numpy(
        affine_mat_after).permute(2, 0, 1).float()
    if use_cuda:
        flow_grid_after = F.affine_grid(
            Variable(affine_mat_after, requires_grad=False).cuda(), feature_tensor.size())
        rotate_feature = F.grid_sample(
            feature_tensor, flow_grid_after, mode="nearest")
    else:
        flow_grid_after = F.affine_grid(
            Variable(affine_mat_after, requires_grad=False), feature_tensor.size())
        rotate_feature = F.grid_sample(
            feature_tensor, flow_grid_after, mode="nearest")
    return rotate_feature


class HERo(nn.Module):
    def __init__(self, n_classes):
        super(HERo, self).__init__()
        self.color_trunk = torchvision.models.resnet101(pretrained=True)
        del self.color_trunk.fc, self.color_trunk.avgpool, self.color_trunk.layer4
        self.depth_trunk = copy.deepcopy(self.color_trunk)

        self.conv1 = nn.Conv2d(2048, 512, 1)
        self.conv2 = nn.Conv2d(512, 128, 1)
        self.conv3 = nn.Conv2d(128, n_classes, 1)

        self.up_2 = nn.Upsample(scale_factor=2, mode="bilinear")

    def forward(self, color, depth):
        # Color
        color_feat_1 = self.color_trunk.conv1(color)  # 3 -> 64
        color_feat_1 = self.color_trunk.bn1(color_feat_1)
        color_feat_1 = self.color_trunk.relu(color_feat_1)
        color_feat_1 = self.color_trunk.maxpool(color_feat_1)
        color_feat_2 = self.color_trunk.layer1(color_feat_1)  # 64 -> 256
        color_feat_3 = self.color_trunk.layer2(color_feat_2)  # 256 -> 512
        color_feat_4 = self.color_trunk.layer3(color_feat_3)  # 512 -> 1024
        # Depth
        depth_feat_1 = self.depth_trunk.conv1(depth)  # 3 -> 64
        depth_feat_1 = self.depth_trunk.bn1(depth_feat_1)
        depth_feat_1 = self.depth_trunk.relu(depth_feat_1)
        depth_feat_1 = self.depth_trunk.maxpool(depth_feat_1)
        depth_feat_2 = self.depth_trunk.layer1(depth_feat_1)  # 64 -> 256
        depth_feat_3 = self.depth_trunk.layer2(depth_feat_2)  # 256 -> 512
        depth_feat_4 = self.depth_trunk.layer3(depth_feat_3)  # 512 -> 1024
        # Concatenate
        x = torch.cat([color_feat_4, depth_feat_4], dim=1)  # 2048
        x = self.conv1(self.up_2(F.relu(x)))
        x = self.conv2(self.up_2(F.relu(x)))
        x = self.conv3(self.up_2(F.relu(x)))
        return x


class reinforcement_net(nn.Module):
    def __init__(self, use_cuda, num_rotations=4):
        super(reinforcement_net, self).__init__()
        self.use_cuda = use_cuda
        self.num_rotations = num_rotations

        # affordance net
        self.grasp_net = HERo(3)

        self.value_net = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(3, 1, 1),
        )
        self.output_prob = None

    def forward(self, input_color_data, input_depth_data, is_volatile=False, specific_rotation=-1, clear_grad=False):
        if is_volatile:  # For choosing action
            output_prob = []
            if self.use_cuda:
                input_color_data = input_color_data.cuda()
                input_depth_data = input_depth_data.cuda()

            # Rotation
            for rotate_idx in range(self.num_rotations):
                theta = np.radians(-90.0+(180.0/self.num_rotations)*rotate_idx)
                rotate_color, rotate_depth = rotate_heightmap(
                    input_color_data, input_depth_data, theta, self.use_cuda)

                with torch.no_grad():
                    affordance = self.grasp_net(rotate_color, rotate_depth)
                    affordance = self.value_net(affordance)
                    output_prob.append(rotate_featuremap(
                        affordance, -theta, self.use_cuda))

            return output_prob

        else:  # For backpropagation, or computing TD target
            self.output_prob = None
            if self.use_cuda:
                input_color_data = input_color_data.cuda()
                input_depth_data = input_depth_data.cuda()

            rotate_idx = specific_rotation
            theta = np.radians(-90.0+(180.0/self.num_rotations)*rotate_idx)
            rotate_color, rotate_depth = rotate_heightmap(
                input_color_data, input_depth_data, theta, self.use_cuda)
            affordance = self.grasp_net(rotate_color, rotate_depth)
            affordance = self.value_net(affordance)

            if clear_grad:
                affordance.detach()

            self.output_prob = rotate_featuremap(
                affordance, -theta, self.use_cuda)

        return self.output_prob


# test model
if __name__ == '__main__':

    import h5py
    import matplotlib.pyplot as plt
    from utils import preprocessing

    hdf5_path = 'Logger05.hdf5'
    f = h5py.File(hdf5_path, "r")

    for key in f.keys():
        group = f[key]
        color = group['state/color'][()]
        depth = group['state/depth'][()]
        pixel_index = group['action'][()]
        reward = group['reward'][()]
        next_color = group['next_state/color'][()]
        next_depth = group['next_state/depth'][()]
        is_empty = group['next_state/empty'][()]
        break
    
    plt.figure()
    plt.imshow(depth)
    plt.show()
    color, depth, padding_width = preprocessing(color, depth)
    plt.figure()
    plt.imshow(depth[0][0])
    plt.show()
    
    print(color.shape)
    print(padding_width)

    rl_net = reinforcement_net(use_cuda=True).cuda()
    rl_net.grasp_net.load_state_dict(
        torch.load("../../weight/HERo_13_objs.pth"))
    out = rl_net(color, depth)
    print(out.shape)
