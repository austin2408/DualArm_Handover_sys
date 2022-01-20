import numpy as np
import torch
import h5py
from torch._C import device

from .trainer import Trainer
from .prioritized_memory import Memory, Transition
from .utils import preprocessing, sample_data, get_action_info, hdf5_to_memory, postProcessing

import warnings
warnings.filterwarnings("ignore")


class Agent():
    def __init__(self, args) -> None:

        self.args = args
        self.iteration = 0
        self.gripper_memory = Memory(args.memory_size)
        self.trainer = Trainer(args)

    def load_pretrained(self, model_path: str) -> None:
        print('[agent] load model ', model_path)
        self.trainer.behavior_net.load_state_dict(torch.load(model_path))
        self.trainer.target_net.load_state_dict(
            self.trainer.behavior_net.state_dict())

    def load_pretrained_graspNet(self, model_path: str) -> None:
        # partially load pretrain from grasp net
        self.trainer.behavior_net.grasp_net.load_state_dict(
            torch.load(model_path))
        self.trainer.target_net.load_state_dict(
            self.trainer.behavior_net.state_dict())

    def set_hdf5_memory(self, hdf5_path='Logger05.hdf5', extra_memory=0) -> None:
        f = h5py.File(hdf5_path, "r")
        self.args.memory_size = len(f.keys()) + extra_memory
        self.gripper_memory = Memory(self.args.memory_size)
        hdf5_to_memory(f, self.gripper_memory)
        print("[agent] load hdf5 to memory: {}".format(
            self.gripper_memory.length))

    def add_transition(self, trans_buf: dict) -> None:

        self.gripper_memory.add(Transition(
            trans_buf['color'],
            trans_buf['depth'],
            trans_buf['pixel_idx'],
            trans_buf['reward'],
            trans_buf['next_color'],
            trans_buf['next_depth'],
            trans_buf['is_empty']
        ))

    def inference(self, color: np.ndarray, depth: np.ndarray):
        color_tensor, depth_tensor, pad = preprocessing(color, depth)
        color_tensor = color_tensor.cuda()
        depth_tensor = depth_tensor.cuda()
        value = self.trainer.behavior_net.forward(
            color_tensor, depth_tensor, is_volatile=True)
        action, value, affordance = postProcessing(
            value, color, depth, color_tensor, pad)

        return action, value, affordance

    def train(self) -> dict:
        record = 0
        loss_list = []
        
        mini_batch, idxs, is_weight = sample_data(
            self.gripper_memory, self.args.mini_batch_size)

        for j in range(len(mini_batch)):
            color = mini_batch[j].color
            depth = mini_batch[j].depth
            pixel_index = mini_batch[j].pixel_idx
            next_color = mini_batch[j].next_color
            next_depth = mini_batch[j].next_depth

            _, rotate_idx = get_action_info(pixel_index)

            reward = mini_batch[j].reward
            if reward > 0:
                record += 1

            td_target = self.trainer.get_label_value(
                reward, next_color, next_depth, mini_batch[j].is_empty)

            if not isinstance(td_target, float):
                td_target = td_target[0]

            loss_ = self.trainer.backprop(
                color, depth, pixel_index, int(td_target), int(is_weight[j]), self.args.mini_batch_size, j == 0, j == int(len(mini_batch)-1))

            loss_list.append(loss_)

        # Update priority
        for j in range(len(mini_batch)):
            color = mini_batch[j].color
            depth = mini_batch[j].depth
            pixel_index = mini_batch[j].pixel_idx
            next_color = mini_batch[j].next_color
            next_depth = mini_batch[j].next_depth
            reward = mini_batch[j].reward

            td_target = self.trainer.get_label_value(
                reward, next_color, next_depth, mini_batch[j].is_empty)

            _, rotate_idx = get_action_info(pixel_index)

            new_value = self.trainer.forward(color, depth, is_volatile=False, specific_rotation=rotate_idx, clear_grad=True)[
                0, pixel_index[1], pixel_index[2]]

            self.gripper_memory.update(idxs[j], td_target-new_value)

        if (self.iteration+1) % self.args.updating_freq == 0:
            self.trainer.target_net.load_state_dict(
                self.trainer.behavior_net.state_dict())

        logs = {'loss mean': sum(loss_list),
                'Success Sample Rate': record/self.args.mini_batch_size,
                }
        self.iteration += 1
        return logs
