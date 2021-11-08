import os
import time
import math
import random
import argparse
import numpy as np


import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable

from test import test
from data_loader import ICDAR2015, Synth80k, ICDAR2013
from mseloss import Maploss

from collections import OrderedDict
from eval.script import getresult
from file_utils import save_final_option, make_logger, AverageMeter
from craft import CRAFT

#3.2768e-5
random.seed(42)


parser = argparse.ArgumentParser(description='CRAFT re-backtime92')

parser.add_argument('--server', default='', type=str, help='server')

parser.add_argument('--results_dir', default='/data/workspace/woans0104/CRAFT-re-backtime92/exp/weekly_back_2', type=str,
                    help='Path to save checkpoints')

parser.add_argument('--synth80k_path', default='/home/data/ocr/detection/SynthText/SynthText', type=str,
                    help='Path to root directory of SynthText dataset')

parser.add_argument('--batch_size', default=128, type = int,
                    help='batch size of training')

parser.add_argument('--epoch', default=128, type = int,
                    help='batch size of training')

parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=0, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--num_workers', default=32, type=int,
                    help='Number of workers used in dataloading')


# for test
def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")

parser.add_argument('--test_dataset', default='icdar2015', type=str, help='test dataset')

parser.add_argument('--text_threshold', default=0.7, type=float, help='text confidence threshold')
parser.add_argument('--low_text', default=0.4, type=float, help='text low-bound score')
parser.add_argument('--link_threshold', default=0.4, type=float, help='link confidence threshold')
parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda to train model')
parser.add_argument('--canvas_size', default=2240, type=int, help='image size for inference')
parser.add_argument('--mag_ratio', default=2, type=float, help='image magnification ratio')
parser.add_argument('--poly', default=False, action='store_true', help='enable polygon type')
parser.add_argument('--show_time', default=False, action='store_true', help='show processing time')
parser.add_argument('--test_folder', default='/home/data/ocr/detection/ICDAR2015/ch4_test_images',
                    type=str, help='folder path to input images for test')
parser.add_argument('--test_folder_gt', default='/home/data/ocr/detection/ICDAR2015/ch4_test_images',
                    type=str, help='folder path to gt text for test')

args = parser.parse_args()


def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")

def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

def adjust_learning_rate(optimizer, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (0.8 ** step)
    print(lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def parse_per_server_dataset(args):

    if args.test_dataset == 'icdar2015':

        if args.server == 'E_server':
            args.synth80k_path = '/data/SynthText'
            args.test_folder = '/data/ICDAR2015/ch4_test_images'
            args.test_folder_gt = '/data/ICDAR2015/ch4_test_localization_transcription_gt'
        elif args.server =='gnnet':
            args.synth80k_path = '/home/data/ocr/detection/SynthText/SynthText'
            args.test_folder = '/home/data/ocr/detection/ICDAR2015/ch4_test_images'
            args.test_folder_gt = '/home/data/ocr/detection/ICDAR2015/ch4_test_localization_transcription_gt'

    elif args.test_dataset == 'icdar2013':

        if args.server == 'E_server':
            args.synth80k_path = '/data/SynthText'
            args.test_folder = '/data/ICDAR2013/Challenge2_Test_Task12_Images'
            args.test_folder_gt = '/data/ICDAR2013/Challenge2_Test_Task1_GT'
        elif args.server == 'gnnet':
            args.synth80k_path = '/home/data/ocr/detection/SynthText/SynthText'
            args.test_folder = '/home/data/ocr/detection/ICDAR2013/Challenge2_Test_Task12_Images'
            args.test_folder_gt = '/home/data/ocr/detection/ICDAR2013/Challenge2_Test_Task1_GT'
    else:
        raise Exception("error not a supported server")

if __name__ == '__main__':

    if not os.path.exists(args.results_dir):
        os.mkdir(args.results_dir)

    parse_per_server_dataset(args)

    dataloader = Synth80k(args.synth80k_path, target_size = 768)
    train_loader = torch.utils.data.DataLoader(
        dataloader,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
        pin_memory=True)


    net = CRAFT(pretrained=True)
    net = torch.nn.DataParallel(net).cuda()
    cudnn.benchmark = True


    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = Maploss()

    #logger
    trn_logger, val_logger = make_logger(path=args.results_dir)

    #save args
    save_final_option(args)

    net.train()


    step_index = 1
    loss_time = 0
    loss_value = 0
    compare_loss = 1
    for epoch in range(args.epoch):
        loss_value = 0

        st = time.time()
        losses = AverageMeter()

        for index, (images, gh_label, gah_label, mask, _) in enumerate(train_loader):
            if index % 20000 == 0 and index != 0:
                step_index += 1
                adjust_learning_rate(optimizer, step_index)


            images = Variable(images.type(torch.FloatTensor)).cuda()
            gh_label = gh_label.type(torch.FloatTensor)
            gah_label = gah_label.type(torch.FloatTensor)
            gh_label = Variable(gh_label).cuda()
            gah_label = Variable(gah_label).cuda()
            mask = mask.type(torch.FloatTensor)
            mask = Variable(mask).cuda()
            # affinity_mask = affinity_mask.type(torch.FloatTensor)
            # affinity_mask = Variable(affinity_mask).cuda()

            out, _ = net(images)

            optimizer.zero_grad()

            out1 = out[:, :, :, 0].cuda()
            out2 = out[:, :, :, 1].cuda()
            loss = criterion(gh_label, gah_label, out1, out2, mask)

            loss.backward()
            optimizer.step()
            losses.update(loss.item(), 1)
            loss_value += loss.item()

            if loss > 1e8 or math.isnan(loss):
                raise Exception("Loss exploded")



            if index % 2 == 0 and index > 0:
                et = time.time()
                print('epoch {}:({}/{}) batch || training time for 2 batch {} || training loss {} ||'
                      .format(epoch, index, len(train_loader), et-st, loss_value/2))
                loss_time = 0
                loss_value = 0
                st = time.time()



            if index % 1000 == 0 and index != 0:
                print('Saving state, index:', index)
                torch.save({
                    'iter': index,
                    'craft': net.module.state_dict(),
                    'optimizer': optimizer.state_dict()
                }, args.results_dir + '/CRAFT_clr_' + repr(index) + '.pth')



                txt_result_path = os.path.join(args.results_dir, 'res_txt')
                test(args, net, txt_result_path)

                try:
                    resdict = getresult(txt_result_path)
                    val_logger.write([index, losses.avg, np.round(resdict['method']['hmean'], 3)])
                except:
                    val_logger.write([index, losses.avg, str(0)])






