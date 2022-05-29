import numpy as np
import scipy.io as sio
import time
import os
import sys
import subprocess
import shlex
import argparse

os.environ['MKL_THREADING_LAYER'] = 'GNU'

# 初始化
parser = argparse.ArgumentParser(description='PyTorch GAN Image Detection')

####################################################################
# Parse command line
####################################################################
parser.add_argument('--gpu-id', default='0', help='gpu id list')
parser.add_argument('--feature', default='image', help='Feature used for training, choose from image and fft')
parser.add_argument('--training_sets', type=str, default=["D_F"], nargs='+',
                    help='Training dataset select from: D_S, D_EM, D_F, D_L and D_S')
parser.add_argument('--test_sets', type=str, default=["D_F"], nargs='+',
                    help='Test dataset select from: horse, zebra, summer, winter, apple, orange, facades, monet, photo, D_S, D_EM, D_F, D_L and D_S')


args = parser.parse_args()

gpu_set = ['0']

if args.feature not in ['image', 'fft']:
    print('Not a valid feature!')
    exit(-1)

for s in args.training_sets:
    if s not in ['D_H', 'D_L', 'D_S', 'D_EM', 'D_F',
                 'D_H_B', 'D_L_B', 'D_S_B', 'D_EM_B', 'D_F_B']:
        print('{} is not a valid training_sets!'.format(s))
        exit(-1)

test_sets_str = ""
for s in args.test_sets:
    if s not in ['horse', 'zebra', 'summer', 'winter', 'apple', 'orange', 'facades', 'monet', 'photo', 'CycleGAN', 'D_L', 'D_S', 'D_EM', 'D_F',
                 'horse_B', 'zebra_B', 'summer_B', 'winter_B', 'apple_B', 'orange_B', 'facades_B', 'monet_B', 'photo_B', 'CycleGAN_B', 'D_L_B', 'D_S_B', 'D_EM_B', 'D_F_B']:
        print('{} is not a valid test_sets!'.format(s))
        exit(-1)
    test_sets_str += s + ' '


number_gpu = len(gpu_set)
process_set = []
index = 0

for dataset in args.training_sets:
    command = 'python ./code/GAN_Detection_Test.py --training_set {}  --test_sets {} --model=resnet --feature {} \
            --gpu-id {} --batch-size=16 --test-batch-size=16 --model_dir ./model_resnet/ --epochs 20'\
            .format(dataset, test_sets_str, args.feature, gpu_set[index % number_gpu])

    print(command)
    p = subprocess.Popen(shlex.split(command))
    process_set.append(p)

    if (index + 1) % number_gpu == 0:
        print('Wait for process end')
        for sub_process in process_set:
            sub_process.wait()

        process_set = []

    time.sleep(10)
    
    for sub_process in process_set:
        sub_process.wait()

