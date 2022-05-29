

from __future__ import division, print_function

import argparse
import csv
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from sklearn.metrics import roc_curve, auc
from torch.autograd import Variable
from torchvision import transforms, models
from tqdm import tqdm

from GAN_Detection_Train import GANDataset
from utils import maybe_cuda

parser = argparse.ArgumentParser(description='GAN Image Detection')

# Training settings
parser.add_argument('--training_set', default='D_H',
                    help='The name of the training set.')
parser.add_argument('--test_sets', default='horse', type=str, nargs='+',
                    help='The list of the test sets.')
parser.add_argument('--feature', default='image',
                    help='Feature used for training, choose from image and fft')
parser.add_argument('--mode', type=int, default=0, 
                    help='fft frequency band, 0: full, 1: low, 2: mid, 3: high')
parser.add_argument('--jpg_level', type=str, default='90',
                    help='Test with different jpg compression effiecients, only effective when use jpg for test set.')
parser.add_argument('--resize_size', type=str, default='200', 
                    help='Test with different resize sizes, only effective when use resize for test set.')

parser.add_argument('--result_dir', default='./final_output/',
                    help='folder to output result in csv')
parser.add_argument('--model_dir', default='./model/',
                    help='folder to output model checkpoints')
parser.add_argument('--model', default='resnet',
                    help='Base classification model')
parser.add_argument('--num-workers', default= 1,
                    help='Number of workers to be created')
parser.add_argument('--pin-memory', type=bool, default=True,
                    help='')
parser.add_argument('--resume', default='', type=str, 
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--start_epoch', default=1, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--epochs', type=int, default=10, 
                    help='number of epochs to train (default: 10)')
parser.add_argument('--batch-size', type=int, default=64, 
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=32,
                    help='input batch size for testing (default: 32)')
parser.add_argument('--lr', type=float, default=0.01, 
                    help='learning rate (default: 0.01)')
parser.add_argument('--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--check_cached', action='store_true', default=True,
                    help='Use cached dataset or not')
parser.add_argument('--seed', type=int, default=-1, 
                    help='random seed (default: -1)')

# Device options
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')


args = parser.parse_args()

suffix = '{}'.format(args.training_set)

if args.feature is not 'image':
    suffix = suffix + '_{}_{}'.format(args.feature, args.mode)

suffix = suffix + '_{}'.format(args.model)

# set the device to use by setting CUDA_VISIBLE_DEVICES env variable in
# order to prevent any memory allocation on unused GPUs
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

args.cuda = not args.no_cuda and torch.cuda.is_available()
if args.cuda:
    cudnn.benchmark = True
    # set random seeds
    if args.seed > -1:
        torch.cuda.manual_seed_all(args.seed)

# set random seeds
if args.seed>-1:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

try:
    os.stat('{}/'.format(args.result_dir))
except:
    os.makedirs('{}/'.format(args.result_dir))

def create_loaders(test_dataset_names):
    kwargs = {'num_workers': args.num_workers, 'pin_memory': args.pin_memory} if args.cuda else {}

    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

    test_loaders = [{'name': name,
                     'dataloader': torch.utils.data.DataLoader(
             GANDataset(train=False,
                     batch_size=args.test_batch_size,
                     name=name,
                     check_cached=args.check_cached,
                     transform=transform),
                        batch_size=args.test_batch_size,
                        shuffle=False, **kwargs)}
                    for name in test_dataset_names]

    return test_loaders


def ro_curve(y_pred, y_label, figure_file, method_name, label):
    fpr, tpr, _ = roc_curve(y_label, y_pred)
    roc_auc = auc(fpr, tpr)
    lw = 2
    plt.cla()
    # plt.plot(fpr, tpr, color='red', label='{} (records:{})'.format(label, len(y_label)))
    plt.plot(fpr, tpr, color='red', label=label+' (area = %0.4f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    # plt.title(method_name + ' (area = %0.4f)' % roc_auc)
    plt.title(method_name)
    fontsize = 14
    plt.xlabel('False Positive Rate', fontsize = fontsize)
    plt.ylabel('True Positive Rate', fontsize = fontsize)
    #plt.title('Receiver Operating Characteristic Curve', fontsize = fontsize)
    plt.legend(loc="lower right")
    plt.savefig(figure_file + ".pdf")
    return


def test(test_loader: object, model, epoch, training_set, test_set, logger_title, logger_label):
    # switch to evaluate mode
    model.eval()

    labels, predicts = [], []
    outputs = []
    pbar = tqdm(enumerate(test_loader))
    for batch_idx, (image_pair, label) in pbar:
        if args.cuda:
            image_pair = image_pair.cuda()
        with torch.no_grad():
            image_pair, label = Variable(image_pair), Variable(label)
        out = model(image_pair)
        _, pred = torch.max(out,1)
        ll = label.data.cpu().numpy().reshape(-1, 1)
        pred = pred.data.cpu().numpy().reshape(-1, 1)
        out = out.data.cpu().numpy().reshape(-1, 2)
        labels.append(ll)
        predicts.append(pred)
        outputs.append(out)

    num_tests = test_loader.dataset.labels.size(0)
    labels = np.vstack(labels).reshape(num_tests)
    predicts = np.vstack(predicts).reshape(num_tests)
    outputs = np.vstack(outputs).reshape(num_tests, 2)
    
    print('\33[91mTest set: {}\n\33[0m'.format(test_set))

    # pos=real=1 neg=fake=0
    pos_label = labels[labels == 1]
    pos_pred = predicts[labels == 1]
    neg_label = labels[labels == 0]
    neg_pred = predicts[labels == 0]

    PREDICT_P = float(pos_pred.shape[0])
    PREDICT_N = float(neg_pred.shape[0])
    P = float(pos_label.shape[0])
    N = float(neg_label.shape[0])

    TP = float(np.sum(pos_label == pos_pred))
    FP = float(PREDICT_P - TP)
    TN = float(np.sum(neg_label == neg_pred))
    FN = float(PREDICT_N - TN)

    # accuracy
    acc = np.sum(labels == predicts) / float(num_tests)
    print('\33[91mTest set: Accuracy: {:.8f}\n\33[0m'.format(acc))

    # TPR
    TPR = TP / P
    print('\33[91mTest set: TPR: {:.8f}\n\33[0m'.format(TPR))

    # TNR neg=fake=0
    TNR = TN / N
    print('\33[91mTest set: TNR: {:.8f}\n\33[0m'.format(TNR))

    # FPR
    FPR = FP / N
    print('\33[91mTest set: FPR: {:.8f}\n\33[0m'.format(FPR))

    # Precision
    PRECISION = TP / (TP + FP)
    print('\33[91mTest set: PRECISION: {:.8f}\n\33[0m'.format(PRECISION))

    # Recall
    RECALL = TP / (TP + FN)
    print('\33[91mTest set: RECALL: {:.8f}\n\33[0m'.format(RECALL))

    # F1-SCORE
    F1_SCORE = 2 * (RECALL * PRECISION) / (RECALL + PRECISION)
    print('\33[91mTest set: F1_SCORE: {:.8f}\n\33[0m'.format(F1_SCORE))

    roc_file = '{}/{}_{}'.format(args.result_dir, suffix, test_set)
    roc_method_name = 'Train:{} | Test:{}'.format(args.training_set, test_set)
    ro_curve(outputs[:, 1], labels, roc_file, roc_method_name, logger_label)

    return acc, PRECISION, RECALL, F1_SCORE

def normal_test(training_set, model):
    checkpoint_url = '{}{}/checkpoint_{}.pth'.format(args.model_dir, suffix, args.epochs);
    if os.path.isfile(checkpoint_url):
        print('\n######## Model checkpoint has been find at {}\n'.format(checkpoint_url))
        checkpoint = torch.load(checkpoint_url)
        model.load_state_dict(checkpoint['state_dict'])
        model = maybe_cuda(model, args.cuda)

        test_loaders = create_loaders(args.test_sets)
        main(training_set, test_loaders, model)


def main(training_set, test_loaders, model):
    # start = args.start_epoch
    # end = start + args.epochs
    test_list, acc_list, precision_list, recall_list, f1_score_list = ['testset'], ['acc'], ['precision'], ['recall'], ['f1_score']
    for test_loader in test_loaders:
        acc, precision, recall, f1_score = test(test_loader['dataloader'], model, 0, training_set,
                                                test_loader['name'], training_set, "ROC")
        test_list.append(str(test_loader['name']))
        acc_list.append(str(acc))
        precision_list.append(str(precision))
        recall_list.append(str(recall))
        f1_score_list.append(str(f1_score))

    with open('{}/{}.csv'.format(args.result_dir, suffix), 'w', newline='') as csvfile:
        score_writer = csv.writer(csvfile, delimiter=',')
        score_writer.writerow(test_list)
        score_writer.writerow(acc_list)
        score_writer.writerow(precision_list)
        score_writer.writerow(recall_list)
        score_writer.writerow(f1_score_list)


if __name__ == '__main__':
    if args.model == 'resnet':
        model = models.resnet34(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)
    elif args.model == 'densenet':
        model = models.densenet121(pretrained=True)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, 2)

    model = maybe_cuda(model, args.cuda)

    normal_test(args.training_set, model)




