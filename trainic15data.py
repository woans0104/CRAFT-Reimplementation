
import os
import time
import math
import random
import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable

import config
from mseloss import Maploss
from data_loader import ICDAR2015, Synth80k, ICDAR2013

from collections import OrderedDict
from file_utils import save_final_option, make_logger, AverageMeter
from craft import CRAFT

random.seed(42)

parser = argparse.ArgumentParser(description='CRAFT re-backtime92')

parser.add_argument('--results_dir', default='', type=str,
                    help='Path to save checkpoints')
parser.add_argument('--synth80k_path', default='/home/data/ocr/detection/SynthText/SynthText', type=str,
                    help='Path to root directory of SynthText dataset')
parser.add_argument('--icdar2015_path', default='/home/data/ocr/detection/ICDAR2015', type=str,
                    help='Path to root directory of icdar2015 dataset')
parser.add_argument("--ckpt_path", default='', type=str,
                    help="path to pretrained model")

parser.add_argument('--epoch', default=1000, type=int,
                    help='epoch')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--eps', default=1e-8, type=float,
                    help='Weight decay for SGD')

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
parser.add_argument('--test_folder_gt', default='/home/data/ocr/detection/ICDAR2015/ch4_test_localization_transcription_gt',
                    type=str, help='folder path to gt text for test')

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


if __name__ == '__main__':

    #import ipdb;ipdb.set_trace()
    #config.EPOCH = args.ckpt_path.split('/')[-1].split('_')[-1].split('.')[0]
    #print(config.EPOCH)

    config.RESULT_DIR = args.results_dir

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

    #print('init model last parameters :{}'.format(net.module.conv_cls[-1].weight.reshape(2, -1)))

    realdata = ICDAR2015(net, args.icdar2015_path, target_size=768)
    real_data_loader = torch.utils.data.DataLoader(
        realdata,
        batch_size=10,
        shuffle=True,
        num_workers=0,
        drop_last=True,
        pin_memory=True)

    optimizer = optim.Adam(net.parameters(), lr=args.lr, eps=args.eps, weight_decay=args.weight_decay)
    criterion = Maploss()


    #logger
    trn_logger, val_logger = make_logger(path=args.results_dir)


    #save args
    save_final_option(args)

    step_index = 0
    loss_time = 0
    loss_value = 0
    compare_loss = 1
    for epoch in range(0,args.epoch):
        train_time_st = time.time()
        loss_value = 0

        # if epoch != 0:
        #     realdata = ICDAR2015(net, args.icdar2015_path, target_size=768)
        #     real_data_loader = torch.utils.data.DataLoader(
        #         realdata,
        #         batch_size=10,
        #         shuffle=True,
        #         num_workers=0,
        #         drop_last=True,
        #         pin_memory=True)

        #5000 iter
        if epoch % 50 == 0 and epoch != 0:
            step_index += 1
            adjust_learning_rate(optimizer, step_index)

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

            if loss > 1e8 or math.isnan(loss):
                trn_logger.write([epoch, 'Loss exploded'])
                raise Exception("Loss exploded")

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



        trn_logger.write([epoch, losses.avg])
        print('Saving state, iter:', epoch)

        torch.save({
            'epoch': epoch,
            'craft': net.module.state_dict(),
            'optimizer': optimizer.state_dict()
        }, args.results_dir + '/CRAFT_clr_' + repr(epoch) + '.pth')

        #
        # txt_result_path = os.path.join(args.results_dir,'res_txt')
        # test(args, net ,txt_result_path)
        # resdict = getresult(txt_result_path)
        # val_logger.write([epoch,np.round(resdict['method']['hmean'],3)])


        








