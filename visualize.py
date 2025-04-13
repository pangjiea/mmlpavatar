
import os
from os import path
import torch
import torch.nn.functional as F
import time
import sys
from argparse import ArgumentParser
import numpy as np

from scene.net_vis import Visualizer
from utils.smpl_utils import init_smpl_pose
from utils.net_utils import net_init, wait_connection

def main(args):
    visualizer = Visualizer()
    init_smpl_pose()
    visualizer.load_model(args.model_dir)
    visualizer.is_send_initial_data = True
    visualizer.gaussians.is_test = True
    visualizer.gaussians.prepare_test()

    net_init(args.ip, args.port)
    while True:
        status = wait_connection()
        visualizer.visualizing()
        if status == 'connected':
            visualizer.is_send_initial_data = True


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Visualizing script parameters")
    parser.add_argument('--model_dir', type=str, default='')
    parser.add_argument('--ip', type=str, default='127.0.0.1')
    parser.add_argument('--port', type=int, default=23456)

    pargs = parser.parse_args(sys.argv[1:])

    print("Visualizing " + pargs.model_dir)
    torch.backends.cuda.matmul.allow_tf32 = True
    main(pargs)