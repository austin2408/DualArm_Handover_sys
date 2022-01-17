import os
import time
import copy
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from scipy import ndimage
from PIL import Image
from torchvision import transforms

from .model import reinforcement_net
from .utils import preprocessing


class Trainer():
    def __init__(self, args):

        self.args = args

        self.behavior_net = reinforcement_net(args.cuda)
        self.target_net = reinforcement_net(args.cuda)
        self.target_net.load_state_dict(
            self.behavior_net.state_dict())
        # Set model to train mode
        self.behavior_net.train()
        self.target_net.train()

        # Huber Loss
        self.criterion = nn.SmoothL1Loss(reduce=False)

        if args.cuda:
            self.behavior_net = self.behavior_net.cuda()
            self.target_net = self.target_net.cuda()
            self.criterion = self.criterion.cuda()
            print("using cuda")

        self.discount_factor = args.discount_factor

        # Initialize optimizer
        self.optimizer = torch.optim.SGD(self.behavior_net.grasp_net.parameters(
        ), lr=args.learning_rate, momentum=0.9, weight_decay=2e-5)

    # Forward pass through image, get Q value

    def forward(self, color_img, depth_img, is_volatile=False, specific_rotation=-1, network="behavior", clear_grad=False):

        input_color_data, input_depth_data, padding_width = preprocessing(
            color_img, depth_img)
        # Pass input data to model
        if network == "behavior":
            output_prob = self.behavior_net.forward(input_color_data, input_depth_data,
                                                    is_volatile=is_volatile, specific_rotation=specific_rotation, clear_grad=clear_grad)
        else:  # Target
            output_prob = self.target_net.forward(input_color_data, input_depth_data,
                                                  is_volatile=is_volatile, specific_rotation=specific_rotation, clear_grad=True)

        lower = int(padding_width/2)
        upper = int(input_color_data.shape[2]/2-padding_width/2)

        if is_volatile == False:
            # only one array
            return output_prob.cpu().detach().numpy()[:, 0, lower:upper, lower:upper]

        for rotate_idx in range(len(output_prob)):
            if rotate_idx == 0:
                grasp_prediction = output_prob[rotate_idx].cpu().detach().numpy()[
                    :, 0, lower:upper, lower:upper]
            else:
                grasp_prediction = np.concatenate((grasp_prediction, output_prob[rotate_idx].cpu(
                ).detach().numpy()[:, 0, lower:upper, lower:upper]))
        return grasp_prediction

    # Get TD target given reward and next state
    def get_label_value(self, reward, next_color, next_depth, is_empty):

        current_reward = reward
        # Compute TD target
        ''' 
        Double DQN 
        TD target = R + discount * Q_target(next state, argmax(Q_behavior(next state, a)))
        '''
        # Use behavior net to find best action in next state
        next_grasp_prediction = self.forward(
            next_color, next_depth, is_volatile=True)

        tmp = np.where(next_grasp_prediction == np.max(next_grasp_prediction))
        next_best_pixel = [tmp[0][0], tmp[1][0], tmp[2][0]]
        rotation = next_best_pixel[0]

        next_prediction = self.forward(
            next_color, next_depth, is_volatile=False, specific_rotation=rotation, network="target")

        future_reward = 0.0
        if not is_empty:
            future_reward = next_prediction[
                0, next_best_pixel[1], next_best_pixel[2]]
        td_target = current_reward + self.discount_factor * future_reward

        del next_prediction
        return td_target

    # Do backwardpropagation
    def backprop(self, color_img, depth_img, action_pix_idx, label_value, is_weight, batch_size, first=False, update=False):

        label = np.zeros((1, 320, 320))
        label[0, action_pix_idx[1]+48, action_pix_idx[2] +
              48] = label_value  # Extra padding
        label_weight = np.zeros((1, 320, 320))
        label_weight[0, action_pix_idx[1]+48,
                     action_pix_idx[2]+48] = 1  # Extra padding
        if first:
            self.optimizer.zero_grad()
        loss_value = 0.0
        out_str = "TD Target: {:.3f}\n Weight: {:.3f}\n".format(
            label_value, is_weight)
        # Forward pass to save gradient
        '''
            0 -> grasp, -90
            1 -> grasp, -45
            2 -> grasp, 0
            3 -> grasp, 45
        '''
        # grasp
        rotation = action_pix_idx[0]
        prediction = self.forward(color_img, depth_img, is_volatile=False,
                                  specific_rotation=rotation, network="behavior", clear_grad=False)
        out_str += "Q: {:.3f}\n".format(
            prediction[0, action_pix_idx[1], action_pix_idx[2]])

        if self.args.cuda:
            loss = self.criterion(self.behavior_net.output_prob.view(1, 320, 320), Variable(torch.from_numpy(label).float().cuda())) * \
                Variable(torch.from_numpy(label_weight).float().cuda(), requires_grad=False) * \
                Variable(torch.from_numpy(np.array([is_weight])).float().cuda(), requires_grad=False) * \
                Variable(torch.from_numpy(
                    np.array([1./batch_size])).float().cuda(), requires_grad=False)
        else:
            loss = self.criterion(self.behavior_net.output_prob.view(1, 320, 320), Variable(torch.from_numpy(label).float())) * \
                Variable(torch.from_numpy(label_weight).float(), requires_grad=False) * \
                Variable(torch.from_numpy(np.array([is_weight])).float(), requires_grad=False) * \
                Variable(torch.from_numpy(
                    np.array([1./batch_size])).float(), requires_grad=False)
        loss = loss.sum()
        loss.backward()
        loss_value = loss.cpu().data.numpy()
        # Grasping is symmetric
        rotation += 4
        prediction = self.forward(color_img, depth_img, is_volatile=False,
                                  specific_rotation=rotation, network="behavior", clear_grad=False)
        out_str += "Q (symmetric): {:.3f}\n".format(
            prediction[0, action_pix_idx[1], action_pix_idx[2]])
        if self.args.cuda:
            loss = self.criterion(self.behavior_net.output_prob.view(1, 320, 320), Variable(torch.from_numpy(label).float().cuda())) * \
                Variable(torch.from_numpy(label_weight).float().cuda(), requires_grad=False) * \
                Variable(torch.from_numpy(np.array([is_weight])).float().cuda(), requires_grad=False) * \
                Variable(torch.from_numpy(
                    np.array([1./batch_size])).float().cuda(), requires_grad=False)
        else:
            loss = self.criterion(self.behavior_net.output_prob.view(1, 320, 320), Variable(torch.from_numpy(label).float())) * \
                Variable(torch.from_numpy(label_weight).float(), requires_grad=False) * \
                Variable(torch.from_numpy(np.array([is_weight])).float(), requires_grad=False) * \
                Variable(torch.from_numpy(
                    np.array([1./batch_size])).float(), requires_grad=False)
        loss = loss.sum()
        loss.backward()
        loss_value += loss.cpu().data.numpy()

        loss_value = loss_value/2

        out_str += "loss: {}".format(loss_value)
        # print(out_str)
        # print('----------------------')

        if update:
            self.optimizer.step()
        return loss_value
