import os
from options.test_options import TestOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import save_images
from util import html

"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
#from __future__ import print_function
#from utils import get_config, get_data_loader_folder, pytorch03_to_pytorch04, load_inception
#from trainer import MUNIT_Trainer, UNIT_Trainer
from torch import nn
from scipy.stats import entropy
import torch.nn.functional as F
#import argparse
#from torch.autograd import Variable
#from data import ImageFolder
import numpy as np
#import torchvision.utils as vutils
try:
    from itertools import izip as zip
except ImportError: # will be 3.x series
    pass
import sys
import torch
import os
from torchvision.models.inception import inception_v3

if __name__ == '__main__':
    opt = TestOptions().parse()
    # hard-code some parameters for test
    opt.num_threads = 1   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True    # no flip
    opt.display_id = -1   # no visdom display
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    model = create_model(opt)
    model.setup(opt)
    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.epoch))
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # pix2pix: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # CycleGAN: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    if opt.eval:
        model.eval()

    if opt.compute_IS:
        IS = []
        all_preds = []
    # if opt.compute_CIS:
    #     CIS = []


    for i, data in enumerate(dataset):
        if i >= opt.num_test:
            break
        model.set_input(data)
        model.test()
        visuals = model.get_current_visuals()
        img_path = model.get_image_paths()
        if i % 5 == 0:
            print('processing (%04d)-th image... %s' % (i, img_path))
        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)

        # 计算IS和CIS
        # 评估指标Inception Score

        # if opt.compute_CIS:
        #     cur_preds = []  # clear cur_preds in each loop

        if opt.compute_IS or opt.compute_IS:
            visuals['fake_B_img']

            # Set up dtype
            if opt.cuda:
                dtype = torch.cuda.FloatTensor
            else:
                if torch.cuda.is_available():
                    print("WARNING: You have a CUDA device, so you should probably set cuda=True")
                dtype = torch.FloatTensor

            # Load the inception networks if we need to compute IS or CIS
            inception = inception_v3(pretrained=True, transform_input=False).type(dtype)
            # freeze the inception models and set eval mode
            inception.eval()
            for param in inception.parameters():
                param.requires_grad = False
            inception_up = nn.Upsample(size=(299, 299), mode='bilinear')

            outputs = visuals['fake_B_img'] # 生成的domainB的假的图片，作为inception的输入

            if opt.compute_IS or opt.compute_CIS:
                pred = F.softmax(inception(inception_up(outputs)), dim=1).cpu().data.numpy()  # get the predicted class distribution
            if opt.compute_IS:
                all_preds.append(pred)
            # if opt.compute_CIS:
            #     cur_preds.append(pred)

            # CIS适用于多模态的评估，在MUNIT中将十种风格作为一组进行append再concatenate
            # 这里没办法，因为只有一张图片。
            # 如下计算会让CIS每次append的都是0,因为py和pyx是一样的！
            # if opt.compute_CIS:
            #     cur_preds = np.concatenate(cur_preds, 0)
            #     py = np.sum(cur_preds, axis=0)  # prior is computed from outputs given a specific input
            #     for j in range(cur_preds.shape[0]):
            #         pyx = cur_preds[j, :]
            #         CIS.append(entropy(pyx, py))


    if opt.compute_IS:
        all_preds = np.concatenate(all_preds, 0)
        py = np.sum(all_preds, axis=0)  # prior is computed from all outputs
        for j in range(all_preds.shape[0]):
            pyx = all_preds[j, :]
            IS.append(entropy(pyx, py))

    if opt.compute_IS:
        print("Inception Score: {}".format(np.exp(np.mean(IS))))
    # if opt.compute_CIS:
    #     print("conditional Inception Score: {}".format(np.exp(np.mean(CIS))))
    # save the website
    webpage.save()

