
import os
import torch
import argparse
import json
import yaml


class Option():
    def __init__(self):
        parser = argparse.ArgumentParser(
            prog="DLP final project", description='This program for offline learning')

        # training hyper parameters
        parser.add_argument("--learning_rate", type=float, default=2.5e-4,
                            help="Learning rate for the trainer, default is 2.5e-4")
        parser.add_argument("--mini_batch_size", type=int, default=10,
                            help="How many transitions should used for learning, default is 10")  # K
        parser.add_argument("--save_freq", type=int, default=100,
                            help="Every how many update should save the model, default is 5")
        parser.add_argument("--updating_freq", type=int, default=10,
                            help="Frequency for updating target network, default is 6")  # C
        parser.add_argument("--iteration", type=int, default=500,
                            help="The train iteration, default is 1000")  # M
        parser.add_argument("--discount_factor", type=float,
                            default=0.9, help="The memory size, default is 0.9")

        parser.add_argument("--memory_size", type=int, default=500,
                            help="The memory size, default is None")

        # save name and load model path
        parser.add_argument("--save_folder", type=str, default=os.getcwd(),
                            help="save model in save folder, default is current path")

        # cuda
        parser.add_argument('--cuda', type=bool, default=True,
                            help='disables CUDA training, default is False')

        self.parser = parser

    def create(self, file=None):
        args = self.parser.parse_args()

        if file:
            with open(file, 'r') as stream:
                data_loaded = yaml.safe_load(stream)
                args.__dict__.update(data_loaded)

        args.cuda = args.cuda and torch.cuda.is_available()
        print(json.dumps(args.__dict__, indent=2, sort_keys=True))
        return args


if __name__ == '__main__':
    args = Option().create("config/offline.yml")
    args = Option().create()
    print(type(args))
