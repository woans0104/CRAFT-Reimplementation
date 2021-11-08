# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
from collections import Iterable



# borrowed from https://github.com/lengstrom/fast-style-transfer/blob/master/src/utils.py
def get_files(img_dir):
    imgs, masks, xmls = list_files(img_dir)
    return imgs, masks, xmls

def list_files(in_path):
    img_files = []
    mask_files = []
    gt_files = []
    for (dirpath, dirnames, filenames) in os.walk(in_path):
        for file in filenames:
            filename, ext = os.path.splitext(file)
            ext = str.lower(ext)
            if ext == '.jpg' or ext == '.jpeg' or ext == '.gif' or ext == '.png' or ext == '.pgm':
                img_files.append(os.path.join(dirpath, file))
            elif ext == '.bmp':
                mask_files.append(os.path.join(dirpath, file))
            elif ext == '.xml' or ext == '.gt' or ext == '.txt':
                gt_files.append(os.path.join(dirpath, file))
            elif ext == '.zip':
                continue
    # img_files.sort()
    # mask_files.sort()
    # gt_files.sort()
    return img_files, mask_files, gt_files

def saveResult_2015(img_file, img, boxes, dirname='./result/', verticals=None, texts=None, gt_file=None):

        """ save text detection result one by one
        Args:
            img_file (str): image file name
            img (array): raw image context
            boxes (array): array of result file
                Shape: [num_detections, 4] for BB output / [num_detections, 4] for QUAD output
        Return:
            None
        """
        img = np.array(img)

        # make result file list
        filename, file_ext = os.path.splitext(os.path.basename(img_file))

        # result directory
        res_file = dirname + "/res_" + filename + '.txt'
        res_img_file = dirname + "/res_" + filename + '.jpg'

        if not os.path.isdir(dirname):
            os.mkdir(dirname)

        with open(res_file, 'w') as f:
            for i, box in enumerate(boxes):

                poly = np.array(box).astype(np.int32).reshape((-1))
                strResult = ','.join([str(p) for p in poly]) + '\r\n'

                f.write(strResult)

                poly = poly.reshape(-1, 2)
                try:
                    cv2.polylines(img, [poly.reshape((-1, 1, 2))], True, color=(0, 255, 0), thickness=2)
                except:
                    pass

                ptColor = (0, 255, 255)
                if verticals is not None:
                    if verticals[i]:
                        ptColor = (255, 0, 0)

                if texts is not None:
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.5
                    cv2.putText(img, "{}".format(texts[i]), (poly[0][0]+1, poly[0][1]+1), font, font_scale, (0, 0, 0), thickness=1)
                    cv2.putText(img, "{}".format(texts[i]), tuple(poly[0]), font, font_scale, (0, 255, 255), thickness=1)


        if gt_file is not None:

            gt_name = "gt_" + filename + '.txt'

            with open(os.path.join(gt_file,gt_name), 'r', encoding="utf8", errors='ignore') as d:
                for l in d.read().splitlines():
                    box = l.split(',')
                    #import ipdb;ipdb.set_trace()

                    box_gt = np.array(list(map(int, box[:8])))
                    gt_poly = box_gt.reshape(-1, 2)
                    gt_poly = np.array(gt_poly).astype(np.int32)

                    if box[-1] == '###':
                        cv2.polylines(img, [gt_poly.reshape((-1, 1, 2))], True, color=(128, 128, 128), thickness=2)
                    else:
                        cv2.polylines(img, [gt_poly.reshape((-1, 1, 2))], True, color=(0, 0, 255), thickness=2)

        #Save result image
        cv2.imwrite(res_img_file, img)


def saveResult(img_file, img, boxes, dirname='./result/', verticals=None, texts=None, gt_file=None,
               dataset='icdar2015'):
        """ save text detection result one by one
        Args:
            img_file (str): image file name
            img (array): raw image context
            boxes (array): array of result file
                Shape: [num_detections, 4] for BB output / [num_detections, 4] for QUAD output
        Return:
            None
        """
        img = np.array(img)

        # make result file list
        filename, file_ext = os.path.splitext(os.path.basename(img_file))

        # result directory
        res_file = dirname + "/res_" + filename + '.txt'
        res_img_file = dirname + "/res_" + filename + '.jpg'

        if not os.path.isdir(dirname):
            os.mkdir(dirname)

        with open(res_file, 'w') as f:
            for i, box in enumerate(boxes):

                if dataset == 'icdar2013':
                    poly = np.array(box).astype(np.int32)
                    min_x = np.min(poly[:,0])
                    max_x = np.max(poly[:,0])
                    min_y = np.min(poly[:,1])
                    max_y = np.max(poly[:,1])
                    strResult = ','.join([str(min_x), str(min_y), str(max_x), str(max_y)]) + '\r\n'

                else: # 'icdar2015'
                    poly = np.array(box).astype(np.int32).reshape((-1))
                    strResult = ','.join([str(p) for p in poly]) + '\r\n'

                f.write(strResult)

                poly = poly.reshape(-1, 2)
                try:
                    cv2.polylines(img, [poly.reshape((-1, 1, 2))], True, color=(0, 255, 0), thickness=2)
                except:
                    pass

                ptColor = (0, 255, 255)
                if verticals is not None:
                    if verticals[i]:
                        ptColor = (255, 0, 0)

                if texts is not None:
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.5
                    cv2.putText(img, "{}".format(texts[i]), (poly[0][0]+1, poly[0][1]+1), font, font_scale, (0, 0, 0), thickness=1)
                    cv2.putText(img, "{}".format(texts[i]), tuple(poly[0]), font, font_scale, (0, 255, 255), thickness=1)


        if gt_file is not None:

            gt_name = "gt_" + filename + '.txt'

            with open(os.path.join(gt_file,gt_name), 'r', encoding="utf8", errors='ignore') as d:
                for l in d.read().splitlines():
                    box = l.split(',')
                    #import ipdb;ipdb.set_trace()
                    if dataset == 'icdar2015':
                        box_gt = np.array(list(map(int, box[:8])))
                        gt_poly = box_gt.reshape(-1, 2)
                        gt_poly = np.array(gt_poly).astype(np.int32)

                        if box[-1] == '###':
                            cv2.polylines(img, [gt_poly.reshape((-1, 1, 2))], True, color=(128, 128, 128), thickness=2)
                        else:
                            cv2.polylines(img, [gt_poly.reshape((-1, 1, 2))], True, color=(0, 0, 255), thickness=2)



                    elif dataset == 'icdar2013':
                        box_gt = np.array(list(map(int, box[:4])))
                        box_gt = np.array(box_gt).astype(np.int32)

                        try:
                            cv2.rectangle(img, (box_gt[0],box_gt[1]),(box_gt[2],box_gt[3]),(0, 0, 255), 3)
                        except:
                            pass


        #Save result image
        cv2.imwrite(res_img_file, img)


def save_final_option(args):

    """ final options """
    with open(f'{args.results_dir}/opt.txt', 'a', encoding="utf-8") as opt_file:
        opt_log = '------------ Options -------------\n'
        arg = vars(args)
        for k, v in arg.items():
            opt_log += f'{str(k)}: {str(v)}\n'
        opt_log += '---------------------------------------\n'
        print(opt_log)
        opt_file.write(opt_log)



def make_logger(path=False):

    # mode = iter or epoch

    def logger_path(path):

        if not os.path.exists(f'{path}'):
            os.mkdir(f'{path}')

        trn_logger_path = os.path.join(f'{path}', f'train.log')
        val_logger_path = os.path.join(f'{path}', f'validation.log')


        return trn_logger_path, val_logger_path


    trn_logger_path, val_logger_path = logger_path( path)

    trn_logger = Logger(trn_logger_path)
    val_logger = Logger(val_logger_path)

    return trn_logger, val_logger


def split_logger(lang_dict):
    eopch_li = []
    loss_li = []
    #acc_li = []
    #ned_li = []

    for i in lang_dict:
        new_dict = i.split()
        eopch_li.append(int(new_dict[0]))
        loss_li.append(float(new_dict[1]))
        #acc_li.append(float(new_dict[2]))
        #ned_li.append(float(new_dict[3]))

    return eopch_li, loss_li

def read_txt(path):
    with open(path, 'r', encoding="utf8", errors='ignore') as d:
        lang_dict = [l for l in d.read().splitlines() if len(l) > 0]

    eopch_li, loss_li = split_logger(lang_dict)

    return eopch_li, loss_li

class Logger(object):
    def __init__(self, path, int_form=':03d', float_form=':.4f'):
        self.path = path
        self.int_form = int_form
        self.float_form = float_form
        self.width = 0

    def __len__(self):
        try: return len(self.read())
        except: return 0

    def write(self, values):
        if not isinstance(values, Iterable):
            values = [values]
        if self.width == 0:
            self.width = len(values)
        assert self.width == len(values), 'Inconsistent number of items.'
        line = ''
        for v in values:
            if isinstance(v, int):
                line += '{{{}}} '.format(self.int_form).format(v)
            elif isinstance(v, float):
                line += '{{{}}} '.format(self.float_form).format(v)
            elif isinstance(v, str):
                line += '{} '.format(v)
            else:
                raise Exception('Not supported type.',v)
        with open(self.path, 'a') as f:
            f.write(line[:-1] + '\n')

    def read(self):
        with open(self.path, 'r') as f:
            log = []
            for line in f:
                values = []
                for v in line.split(' '):
                    try:
                        v = float(v)
                    except:
                        pass
                    values.append(v)
                log.append(values)
        return log





class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.sum_2 = 0 # sum of squares
        self.count = 0
        self.std = 0

    def update(self, val, n=1):
        if val!=None: # update if val is not None
            self.val = val
            self.sum += val * n
            self.sum_2 += val**2 * n
            self.count += n
            self.avg = self.sum / self.count
            self.std = np.sqrt(self.sum_2/self.count - self.avg**2)

        else:
            pass


