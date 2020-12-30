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
from tqdm import tqdm
from torch.autograd import Variable
from scipy import linalg

def read_stats_file(filepath):
    """read mu, sigma from .npz"""
    if filepath.endswith('.npz'):
        f = np.load(filepath)
        m, s = f['mu'][:], f['sigma'][:]
        f.close()
    else:
        raise Exception('ERROR! pls pass in correct npz file %s' % filepath)
    return m, s

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths %s, %s' % (mu1.shape, mu2.shape)
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions %s, %s' % (sigma1.shape, sigma2.shape)
    diff = mu1 - mu2
    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real
    tr_covmean = np.trace(covmean)
    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

class ScoreModel:
    def __init__(self, mode, cuda=True,
                 stats_file='', mu1=0, sigma1=0):
        """
        Computes the inception score of the generated images
            cuda -- whether or not to run on GPU
            mode -- image passed in inceptionV3 is normalized by mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
                and in range of [-1, 1]
                1: image passed in is normalized by mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                2: image passed in is normalized by mean=[0.500, 0.500, 0.500], std=[0.500, 0.500, 0.500]
        """
        # load mu, sigma for calc FID
        self.calc_fid = False
        if stats_file:
            self.calc_fid = True
            self.mu1, self.sigma1 = read_stats_file(stats_file)
        elif type(mu1) == type(sigma1) == np.ndarray:
            self.calc_fid = True
            self.mu1, self.sigma1 = mu1, sigma1

        # Set up dtype
        if cuda:
            self.dtype = torch.cuda.FloatTensor
        else:
            if torch.cuda.is_available():
                print("WARNING: You have a CUDA device, so you should probably set cuda=True")
            self.dtype = torch.FloatTensor

        # setup image normalization mode
        self.mode = mode
        if self.mode == 1:
            transform_input = True
        elif self.mode == 2:
            transform_input = False
        else:
            raise Exception("ERR: unknown input img type, pls specify norm method!")
        self.inception_model = inception_v3(pretrained=True, transform_input=transform_input).type(self.dtype)
        self.inception_model.eval()
        # self.up = nn.Upsample(size=(299, 299), mode='bilinear', align_corners=False).type(self.dtype)

        # remove inception_model.fc to get pool3 output 2048 dim vector
        self.fc = self.inception_model.fc
        self.inception_model.fc = nn.Sequential()

        # wrap with nn.DataParallel
        self.inception_model = nn.DataParallel(self.inception_model)
        self.fc = nn.DataParallel(self.fc)

    def __forward(self, x):
        """
        x should be N x 3 x 299 x 299
        and should be in range [-1, 1]
        """
        x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        x = self.inception_model(x)
        pool3_ft = x.data.cpu().numpy()

        x = self.fc(x)
        preds = F.softmax(x, 1).data.cpu().numpy()
        return pool3_ft, preds

    @staticmethod
    def __calc_is(preds, n_split, return_each_score=False):
        """
        regularly, return (is_mean, is_std)
        if n_split==1 and return_each_score==True:
            return (scores, 0)
            # scores is a list with len(scores) = n_img = preds.shape[0]
        """

        n_img = preds.shape[0]
        # Now compute the mean kl-div
        split_scores = []
        for k in range(n_split):
            part = preds[k * (n_img // n_split): (k + 1) * (n_img // n_split), :]
            py = np.mean(part, axis=0)
            scores = []
            for i in range(part.shape[0]):
                pyx = part[i, :]
                scores.append(entropy(pyx, py))
            split_scores.append(np.exp(np.mean(scores)))
            if n_split == 1 and return_each_score:
                return scores, 0
        return np.mean(split_scores), np.std(split_scores)

    @staticmethod
    def __calc_stats(pool3_ft):
        mu = np.mean(pool3_ft, axis=0)
        sigma = np.cov(pool3_ft, rowvar=False)
        return mu, sigma

    def get_score_image_tensor(self, imgs_nchw, mu1=0, sigma1=0,
                               n_split=10, batch_size=1, return_stats=False,
                               return_each_score=False):
        """
        param:
            imgs_nchw -- Pytorch Tensor, size=(N,C,H,W), in range of [-1, 1]
            batch_size -- batch size for feeding into Inception v3
            n_splits -- number of splits
        return:
            is_mean, is_std, fid
            mu, sigma of dataset

            regularly, return (is_mean, is_std)
            if n_split==1 and return_each_score==True:
                return (scores, 0)
                # scores is a list with len(scores) = n_img = preds.shape[0]
        """

        n_img = imgs_nchw.shape[0]

        assert batch_size > 0
        assert n_img > batch_size

        pool3_ft = np.zeros((n_img, 2048))
        preds = np.zeros((n_img, 1000))
        for i in tqdm(range(np.int32(np.ceil(1.0 * n_img / batch_size)))):
            batch_size_i = min((i+1) * batch_size, n_img) - i * batch_size
            batchv = Variable(imgs_nchw[i * batch_size:i * batch_size + batch_size_i, ...].type(self.dtype))
            pool3_ft[i * batch_size:i * batch_size + batch_size_i], preds[i * batch_size:i * batch_size + batch_size_i] = self.__forward(batchv)

        # if want to return stats
        # or want to calc fid
        if return_stats or \
                type(mu1) == type(sigma1) == np.ndarray or self.calc_fid:
            mu2, sigma2 = self.__calc_stats(pool3_ft)

        if self.calc_fid:
            mu1 = self.mu1
            sigma1 = self.sigma1

        is_mean, is_std = self.__calc_is(preds, n_split, return_each_score)

        fid = -1
        if type(mu1) == type(sigma1) == np.ndarray or self.calc_fid:
            fid = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)

        if return_stats:
            return is_mean, is_std, fid, mu2, sigma2
        else:
            return is_mean, is_std, fid

    def get_score_dataset(self, dataset, mu1=0, sigma1=0,
                          n_split=10, batch_size=32, return_stats=False,
                          return_each_score=False):
        """
        get score from a dataset
        param:
            dataset -- pytorch dataset, img in range of [-1, 1]
            batch_size -- batch size for feeding into Inception v3
            n_splits -- number of splits
        return:
            is_mean, is_std, fid
            mu, sigma of dataset

            regularly, return (is_mean, is_std)
            if n_split==1 and return_each_score==True:
                return (scores, 0)
                # scores is a list with len(scores) = n_img = preds.shape[0]
        """

        n_img = len(dataset)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

        pool3_ft = np.zeros((n_img, 2048))
        preds = np.zeros((n_img, 1000))
        for i, batch in tqdm(enumerate(dataloader, 0)):
            batch = batch.type(self.dtype)
            batchv = Variable(batch)
            batch_size_i = batch.size()[0]
            pool3_ft[i * batch_size:i * batch_size + batch_size_i], preds[i * batch_size:i * batch_size + batch_size_i] = self.__forward(batchv)

        # if want to return stats
        # or want to calc fid
        if return_stats or \
                type(mu1) == type(sigma1) == np.ndarray or self.calc_fid:
            mu2, sigma2 = self.__calc_stats(pool3_ft)

        if self.calc_fid:
            mu1 = self.mu1
            sigma1 = self.sigma1

        is_mean, is_std = self.__calc_is(preds, n_split, return_each_score)

        fid = -1
        if type(mu1) == type(sigma1) == np.ndarray or self.calc_fid:
            fid = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)

        if return_stats:
            return is_mean, is_std, fid, mu2, sigma2
        else:
            return is_mean, is_std, fid

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
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.epoch))
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    if opt.eval:
        model.eval()

    if opt.compute_fid:
        img_list_tensor1 = []
        img_list_tensor2 = []
        IS = []
        all_preds = []

    for i, data in enumerate(dataset):
        if i >= opt.num_test:
            break
        model.set_input(data)
        model.test()
        visuals = model.get_current_visuals()
        img_path = model.get_image_paths()
        # get AtoB
        is_AtoB = model.get_AtoB()
        if i % 5 == 0:
            print('processing (%04d)-th image... %s' % (i, img_path))
        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)

        # 计算IS和FID
        if opt.compute_fid:
            if opt.model=='cycle_gan':
                is_fid_model = ScoreModel(mode=2, cuda=True)
                img_list_tensor1.append(visuals['real_A']) if is_AtoB else img_list_tensor1.append(
                    visuals['real_B'])
                img_list_tensor2.append(visuals['fake_A']) if is_AtoB else img_list_tensor2.append(
                    visuals['fake_B'])
            elif opt.model=='objectvaried_gan':
                if opt.netG == 'star':
                    is_fid_model = ScoreModel(mode=2, cuda=True)
                    img_list_tensor1.append(visuals['real_A_img']) if is_AtoB else img_list_tensor1.append(
                        visuals['real_B_img'])
                    img_list_tensor2.append(visuals['fake_A_img']) if is_AtoB else img_list_tensor2.append(
                        visuals['fake_B_img'])
                else:
                    is_fid_model = ScoreModel(mode=2, cuda=True)
                    img_list_tensor1.append(visuals['real_A_img']) if is_AtoB else img_list_tensor1.append(
                        visuals['real_B_img'])
                    img_list_tensor2.append(visuals['fake_A_img']) if is_AtoB else img_list_tensor2.append(visuals['fake_B_img'])
            else:
                pass







    img_list_tensor1 = torch.cat(img_list_tensor1, dim=0)
    img_list_tensor2 = torch.cat(img_list_tensor2, dim=0)

    if opt.compute_fid:
        print('Calculating 1st stat ...')
        is_mean1, is_std1, _, mu1, sigma1 = \
            is_fid_model.get_score_image_tensor(img_list_tensor1, n_split=10, return_stats=True)

        print('Calculating 2nd stat ...')
        is_mean2, is_std2, fid = is_fid_model.get_score_image_tensor(img_list_tensor2,
                                                                     mu1=mu1, sigma1=sigma1,
                                                                     n_split=10)

        print('1st IS score =', is_mean1, ',', is_std1)
        print('2nd IS score =', is_mean2, ',', is_std2)
        print('FID =', fid)

    # save the website
    webpage.save()

