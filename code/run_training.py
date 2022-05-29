import numpy as np
import scipy.io as sio
import time
import os
import sys
import subprocess
import shlex
import argparse

os.environ['MKL_THREADING_LAYER'] = 'GNU'

# Add by Holly
parser = argparse.ArgumentParser()

####################################################################
# Parse command line
####################################################################
parser.add_argument('--feature', default='image', help='Feature used for training, choose from image and fft')
parser.add_argument('--gpu-id', default='0', help='gpu id list')
parser.add_argument('--training_sets', type=str, default='D_S', nargs='+',
                    help='Training dataset select from: D_S, D_EM, D_F, D_L and D_S')
parser.add_argument('--test_sets', type=str, default='D_S', nargs='+',
                    help='Test dataset select from: horse, zebra, summer, winter, apple, orange, facades, monet, photo, D_S, D_EM, D_F, D_L and D_S')
parser.add_argument('--with_balance', type=bool, default=False, help='Compare datasets with and without balance')

args = parser.parse_args()

gpu_set = args.gpu_id.split(',')
# add more gpu if wanted
# gpu_set = ['0','1']

number_gpu = len(gpu_set)

if args.feature not in ['image', 'fft']:
    print('Not a valid feature!')
    exit(-1)

for s in args.training_sets:
    if s not in ['D_H', 'D_L', 'D_S', 'D_EM', 'D_F']:
        print('Not a valid training_sets!')
        exit(-1)

for s in args.test_sets:
    if s not in ['horse', 'zebra', 'summer', 'winter', 'apple', 'orange', 'facades', 'monet', 'photo', 'CycleGAN', 'D_H', 'D_L', 'D_S', 'D_EM', 'D_F']:
        print('Not a valid test_sets!')
        exit(-1)

process_set = []
index = 0

if args.with_balance is True:
    balance_suffix_list = ['_B', '']
else:
    balance_suffix_list = ['']

for suf in balance_suffix_list:
    for dataset in args.training_sets:
        param_training_set = dataset + suf
        param_test_sets_str = ""
        for s in args.test_sets:
            param_test_sets_str += s + suf + ' '

        command = 'python ./code/GAN_Detection_Train.py --training_set {} --test_sets {} --model=resnet --feature {} \
                    --gpu-id {} --batch-size 16 --test-batch-size 16 --model_dir ./model_resnet/  --log_dir ./resnet_log/ \
                    --enable-logging False --epochs 20 ' \
            .format(param_training_set, param_test_sets_str, args.feature, gpu_set[index % number_gpu])

        print(command)
        p = subprocess.Popen(shlex.split(command))
        process_set.append(p)

        if (index + 1) % number_gpu == 0:
            print('Wait for process end')
            for sub_process in process_set:
                sub_process.wait()

            process_set = []
        index += 1
        time.sleep(60)

    for sub_process in process_set:
        sub_process.wait()

