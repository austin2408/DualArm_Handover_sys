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


class HANet(nn.Module):
    def __init__(self, n_classes):
        super(HANet, self).__init__()
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
