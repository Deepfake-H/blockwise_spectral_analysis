
import argparse
import os
import shutil
import random


parser = argparse.ArgumentParser(description='Balance data from directions')

# Training settings
parser.add_argument('--datasets', type=str, default='D_S', nargs='+',
                    help='dataset select from: horse, zebra, summer, winter, apple, orange, facades, monet, photo, D_L, D_S, D_EM, D_F ')
parser.add_argument('--input', default='./data',
                    help='path to Output')
parser.add_argument('--output', default='./data',
                    help='path to Output')
parser.add_argument('--output_suffix', default='B',
                    help='path to Output')
parser.add_argument('--random', default=True,
                    help='Random select or not')

parser.add_argument('--merge', default=False,
                    help='merge datasets or not')
parser.add_argument('--merge_name', default="CycleGAN",
                    help='merge dataset name')


args = parser.parse_args()


def main():
    for dataset in args.datasets:
        process_one_dir(dataset, "trainA")
        if dataset != 'D_H':
            process_one_dir(dataset, "testA")


def copy_file(source_path, target_path, number_of_file, is_random=False, prefixes=""):
    if not os.path.exists(target_path):
        os.makedirs(target_path)

    if os.path.exists(source_path):
        for root, dirs, files in os.walk(source_path):
            idx = 0
            if is_random:
                random.shuffle(files)
            for file in files:
                src_file = os.path.join(root, file)
                tar_file = os.path.join(target_path, prefixes + file)
                shutil.copy(src_file, tar_file)
                idx += 1
                if idx >= number_of_file:
                    break
            print('{} files copied from [{}] to [{}] (Random:{})'.format(idx, source_path, target_path, is_random))


def process_one_dir(dataset, sub_folder):
    real_dir = '{}/real/{}/{}/'.format(args.input, dataset, sub_folder)
    fake_dir = '{}/fake/{}/{}/'.format(args.input, dataset, sub_folder)

    output_real_dir = '{}/real/{}_{}/{}/'.format(args.output, dataset, args.output_suffix, sub_folder)
    output_fake_dir = '{}/fake/{}_{}/{}/'.format(args.output, dataset, args.output_suffix, sub_folder)

    real_file_nums = sum([len(files) for root, dirs, files in os.walk(real_dir)])
    fake_file_nums = sum([len(files) for root, dirs, files in os.walk(fake_dir)])

    num_to_copy_real = min(real_file_nums, fake_file_nums)
    num_to_copy_fake = num_to_copy_real

    prefixes = ""
    if args.merge:
        output_real_dir = '{}/real/{}/{}/'.format(args.output, args.merge_name, sub_folder)
        output_fake_dir = '{}/fake/{}/{}/'.format(args.output, args.merge_name, sub_folder)
        prefixes = dataset
        num_to_copy_real = real_file_nums
        num_to_copy_fake = fake_file_nums

    print('### Processing dataset: {}/{} (real:{} - fake:{})'.format(dataset, sub_folder, real_file_nums, fake_file_nums))
    copy_file(real_dir, output_real_dir, num_to_copy_real, args.random, prefixes)
    copy_file(fake_dir, output_fake_dir, num_to_copy_fake, args.random, prefixes)
    print('')


if __name__ == '__main__':
    main()
