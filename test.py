"""  
Copyright (c) 2019-present NAVER Corp.
MIT License
"""

# -*- coding: utf-8 -*-

import os
import time
import argparse

import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

import cv2
import numpy as np
import imgproc


from craft import CRAFT
from collections import OrderedDict
from eval.script import eval_2015
import craft_utils
import file_utils


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

def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")



#result_folder = '/data/CRAFT-pytorch/result/'
# if not os.path.isdir(args.result_folder):
#     os.mkdir(args.result_folder)



def test_net(args, net, image, text_threshold, link_threshold, low_text, cuda, poly):

    cuda =True
    t0 = time.time()
    torch.cuda.empty_cache()

    with torch.no_grad():
        # resize
        img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, args.canvas_size,
                                                                              interpolation=cv2.INTER_LINEAR,
                                                                              mag_ratio=args.mag_ratio)
        ratio_h = ratio_w = 1 / target_ratio

        # preprocessing
        x = imgproc.normalizeMeanVariance(img_resized)
        x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
        x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
        if cuda:
            x = Variable(x.cuda())

        # forward pass
        y, _ = net(x)


        # make score and link map
        score_text = y[0,:,:,0].cpu().data.numpy()
        score_link = y[0,:,:,1].cpu().data.numpy()

        t0 = time.time() - t0
        t1 = time.time()

        # Post-processing
        boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

        # coordinate adjustment
        boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
        polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
        for k in range(len(polys)):
            if polys[k] is None: polys[k] = boxes[k]

        t1 = time.time() - t1

        # render results (optional)
        render_img = score_text.copy()
        render_img = np.hstack((render_img, score_link))
        ret_score_text = imgproc.cvt2HeatmapImg(render_img)

        if args.show_time : print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))

    return boxes, polys, ret_score_text



def test(args, modelpara, result_folder):
    if not os.path.isdir(result_folder):
        os.mkdir(result_folder)


    """ For test images in a folder """
    image_list, _, _ = file_utils.get_files(args.test_folder)


    if type(modelpara) is str:
        # load net
        net = CRAFT()     # initialize
        print('Loading weights from checkpoint {}'.format(modelpara))

        net_param = torch.load(modelpara)

        try:
            if args.cuda:
                net.load_state_dict(copyStateDict(net_param['craft']))
                net = net.cuda()
                net = torch.nn.DataParallel(net)
                cudnn.benchmark = False

            else:
                net.load_state_dict(copyStateDict(torch.load(net_param['craft'], map_location='cpu')))
        except:
            net.load_state_dict(copyStateDict(net_param))
            net = net.cuda()
            net = torch.nn.DataParallel(net)
            cudnn.benchmark = False

    else:
        net = modelpara
    net.eval()

    t = time.time()

    # load data
    for k, image_path in enumerate(image_list):
        print("Test image {:d}/{:d}: {:s}".format(k+1, len(image_list), image_path), end='\r')
        image = imgproc.loadImage(image_path)

        bboxes, polys, score_text = test_net(args, net, image, args.text_threshold, args.link_threshold, args.low_text,
                                             args.cuda, args.poly)


        # save score text
        filename, file_ext = os.path.splitext(os.path.basename(image_path))
        mask_file = result_folder + "/res_" + filename + '_mask.jpg'
        cv2.imwrite(mask_file, score_text)



        file_utils.saveResult_2015(image_path, image[:,:,::-1], polys, dirname=result_folder,
                                   gt_file=args.test_folder_gt, dataset=args.test_dataset)



    print("elapsed time : {}s".format(time.time() - t))




if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='CRAFT Text Detection')
    parser.add_argument('--ckpt_path', default='weights/craft_mlt_25k.pth', type=str, help='pretrained model')
    parser.add_argument('--text_threshold', default=0.7, type=float, help='text confidence threshold')
    parser.add_argument('--low_text', default=0.4, type=float, help='text low-bound score')
    parser.add_argument('--link_threshold', default=0.4, type=float, help='link confidence threshold')
    parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda to train model')
    parser.add_argument('--canvas_size', default=2240, type=int, help='image size for inference')
    parser.add_argument('--mag_ratio', default=2, type=float, help='image magnification ratio')
    parser.add_argument('--poly', default=False, type=str2bool, help='enable polygon type')
    parser.add_argument('--show_time', default=False, action='store_true', help='show processing time')
    parser.add_argument('--results_dir', default='',
                        type=str, help='Path to save checkpoints')
    parser.add_argument('--test_folder', default='/home/data/ocr/detection/ICDAR2015/ch4_test_images',
                        type=str, help='folder path to input images')
    parser.add_argument('--test_folder_gt', default=None,
                        type=str, help='folder path to gt text for test')
    parser.add_argument('--test_dataset', default='', type=str, help='test dataset')

    args = parser.parse_args()

    arg = vars(args)


    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)

    #save args
    file_utils.save_final_option(args)


    test(args, modelpara=args.ckpt_path, result_folder=args.results_dir)
    resdict = eval_2015(args.results_dir)

    import json
    resdict = json.dumps(resdict["method"])
    with open(f'{args.results_dir}/result_score.txt', 'a', encoding="utf-8") as opt_file:
        opt_file.write(resdict)
    opt_file.close()


