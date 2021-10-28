import os
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np
import scipy.io as scio
import argparse
import time
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
import random
import h5py
import re
from test import test


from math import exp
from data_loader import ICDAR2015, Synth80k, ICDAR2013
from mseloss import Maploss



from collections import OrderedDict
from eval.script import getresult
from craft_utils import save_final_option, make_logger, AverageMeter


from PIL import Image
from torchvision.transforms import transforms
from craft import CRAFT
from torch.autograd import Variable
from multiprocessing import Pool

#3.2768e-5
random.seed(42)

# class SynAnnotationTransform(object):
#     def __init__(self):
#         pass
#     def __call__(self, gt):
#         image_name = gt['imnames'][0]
parser = argparse.ArgumentParser(description='CRAFT reimplementation')


parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')

parser.add_argument('--results_dir', default='/data/workspace/woans0104/CRAFT-re-backtime92/exp/weekly_back_2', type=str,
                    help='Path to save checkpoints')

parser.add_argument('--synth80k_path', default='/home/data/ocr/detection/SynthText/SynthText', type=str,
                    help='Path to root directory of SynthText dataset')
parser.add_argument('--icdar2015_path', default='/home/data/ocr/detection/ICDAR2015', type=str,
                    help='Path to root directory of SynthText dataset')
parser.add_argument("--ckpt_path", default='/data/workspace/woans0104/CRAFT-reimplementation/exp'
                                           '/ICDAR2015_test/unof-Syndata.pth', type=str,
                    help="path to pretrained model")



#parser.add_argument('--cdua', default=True, type=str2bool,
                    #help='Use CUDA to train model')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')



args = parser.parse_args()






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

def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (0.8 ** step)
    print(lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':


    if not os.path.exists(args.results_dir):
        os.mkdir(args.results_dir)



    dataloader = Synth80k(args.synth80k_path, target_size = 768)
    train_loader = torch.utils.data.DataLoader(
        dataloader,
        batch_size=2,
        shuffle=True,
        num_workers=0,
        drop_last=True,
        pin_memory=True)
    batch_syn = iter(train_loader)


    net = CRAFT()


    net_param = torch.load(args.ckpt_path)
    try:
        net.load_state_dict(copyStateDict(net_param['craft']))
    except:
        net.load_state_dict(copyStateDict(net_param))


    net = net.cuda()
    net = torch.nn.DataParallel(net).cuda()
    cudnn.benchmark = True



    realdata = ICDAR2015(net, args.icdar2015_path, target_size=768)
    real_data_loader = torch.utils.data.DataLoader(
        realdata,
        batch_size=10,
        shuffle=True,
        num_workers=0,
        drop_last=True,
        pin_memory=True)


    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = Maploss()
    #criterion = torch.nn.MSELoss(reduce=True, size_average=True)



    #logger
    trn_logger, val_logger = make_logger(path=args.results_dir)


    #save args
    save_final_option(args)

    step_index = 0



    loss_time = 0
    loss_value = 0
    compare_loss = 1
    for epoch in range(1000):
        train_time_st = time.time()
        loss_value = 0

        if epoch != 0:
            realdata = ICDAR2015(net, args.icdar2015_path, target_size=768)
            real_data_loader = torch.utils.data.DataLoader(
                realdata,
                batch_size=10,
                shuffle=True,
                num_workers=0,
                drop_last=True,
                pin_memory=True)


        if epoch % 50 == 0 and epoch != 0:
            step_index += 1
            adjust_learning_rate(optimizer, args.gamma, step_index)

        if not net.training:
            net.train()

        print('train mode :',net.training)
        st = time.time()
        losses = AverageMeter()
        for index, (real_images, real_gh_label, real_gah_label, real_mask, _) in enumerate(real_data_loader):

            #real_images, real_gh_label, real_gah_label, real_mask = next(batch_real)
            syn_images, syn_gh_label, syn_gah_label, syn_mask, __ = next(batch_syn)
            images = torch.cat((syn_images,real_images), 0)
            gh_label = torch.cat((syn_gh_label, real_gh_label), 0)
            gah_label = torch.cat((syn_gah_label, real_gah_label), 0)
            mask = torch.cat((syn_mask, real_mask), 0)
            #affinity_mask = torch.cat((syn_mask, real_affinity_mask), 0)


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
            loss_value += loss.item()
            losses.update(loss.item(), images.size(0))

            if index % 2 == 0 and index > 0:
                et = time.time()
                print('epoch {}:({}/{}) batch || training time for 2 batch {} || training loss {} ||'
                      .format(epoch, index, len(real_data_loader), et-st, loss_value/2))
                loss_time = 0
                loss_value = 0
                st = time.time()
            # if loss < compare_loss:
            #     print('save the lower loss iter, loss:',loss)
            #     compare_loss = loss
            #     torch.save(net.module.state_dict(),
            #                '/data/CRAFT-pytorch/real_weights/lower_loss.pth')

        trn_logger.write([epoch, losses.avg])
        print('Saving state, iter:', epoch)
        torch.save(net.module.state_dict(),
                args.results_dir+'/CRAFT_clr_' + repr(epoch) + '.pth')


        txt_result_path = os.path.join(args.results_dir,'res_txt')
        test(args.results_dir+'/CRAFT_clr_' + repr(epoch) + '.pth',txt_result_path )
        resdict = getresult(txt_result_path)

        val_logger.write([epoch,np.round(resdict['method']['hmean'],3)])
        








