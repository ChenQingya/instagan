import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import numpy as np
import copy

from contextualloss.contextual_loss import Contextual_Loss
import torch.nn.functional as F

from torchvision import transforms
from PIL import Image
from util import util
import os

class InstaGANModel(BaseModel):
    def name(self):
        return 'InstaGANModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        # default CycleGAN did not use dropout，instagan也不使用dropout
        parser.set_defaults(no_dropout=True)
        parser.add_argument('--set_order', type=str, default='decreasing', help='order of segmentation')			# why there is a order for segmentation?
        parser.add_argument('--ins_max', type=int, default=4, help='maximum number of instances to forward')		# 所有训练中，最多可以使用到的instance的数目
        parser.add_argument('--ins_per', type=int, default=2, help='number of instances to forward, for one pass')	# 一次迭代中，使用到的instance的数目
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_idt', type=float, default=1.0, help='use identity mapping. Setting lambda_idt other than 0 has an effect of scaling the weight of the identity mapping loss')
            # what's the difference between ctx and idt?
            parser.add_argument('--lambda_ctx', type=float, default=1.0, help='use context preserving. Setting lambda_ctx other than 0 has an effect of scaling the weight of the context preserving loss')

        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        self.ins_iter = self.opt.ins_max // self.opt.ins_per  				# number of forward iteration, self.ins_iter=4//2，所以self.ins_iter=2
                                                                            # “//”，在python中，整数除法，这个叫“地板除”，3//2=1

        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        #self.loss_names = ['D_A', 'G_A', 'cyc_A', 'idt_A', 'ctx_A', 'D_B', 'G_B', 'cyc_B', 'idt_B', 'ctx_B']
        #self.loss_names = ['D_A', 'G_A', 'cyc_A', 'idt_A', 'ctx_A', 'cx_A', 'D_B', 'G_B', 'cyc_B', 'idt_B', 'ctx_B', 'cx_B']
        self.loss_names = ['D_A', 'G_A', 'cyc_A', 'idt_A', 'D_B', 'G_B', 'cyc_B', 'idt_B']

        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        visual_names_A_img = ['real_A_img', 'fake_B_img', 'rec_A_img']
        visual_names_B_img = ['real_B_img', 'fake_A_img', 'rec_B_img']
        visual_names_A_seg = ['real_A_seg', 'fake_B_seg', 'rec_A_seg']
        visual_names_B_seg = ['real_B_seg', 'fake_A_seg', 'rec_B_seg']
        self.visual_names = visual_names_A_img + visual_names_A_seg + visual_names_B_img + visual_names_B_seg

        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:													#isTrain：True时表示是执行了train.py，否则执行了test.py
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']					#isTrain为True时，保存生成器和判别器
        else:
            self.model_names = ['G_A', 'G_B']								#isTrain为False时，只保存生成器

        # load/define networks
        # The naming conversion is different from those used in the paper
        # Code (paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)	# opt.norm默认是'instance'
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        if self.isTrain:
            use_sigmoid = opt.no_lsgan	#why?为什么不使用lsgan，就表示用sigmoid？
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            self.fake_A_pool = ImagePool(opt.pool_size)	# '--pool_size', type=int, default=50, help='the size of image buffer that stores previously generated images'
            self.fake_B_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan).to(self.device)	# 通过opt.no_lsgan控制，使用MSEloss或者BSEloss
            self.criterionCyc = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()

            # 以下初始化optimizer涉及两个函数，filter()和lambda
            # filter() 函数
            # 	用于过滤序列，过滤掉不符合条件的元素，返回由符合条件元素组成的新列表。
            # 	该接收两个参数，第一个为函数，第二个为序列，序列的每个元素作为参数传递给函数进行判，然后返回 True 或 False，最后将返回 True 的元素放到新列表中。python3中filter返回迭代器对象
            # lambda p: p.requires_grad
            # 	这里匿名函数，p是参数，p.required_grad是表达式

            # initialize optimizers
            # 这里的filter，第一个为函数（匿名函数），第二个为序列（包含netG_A和netG_B的所有parameter），返回这些parameter中符合requires_grad=True的parameter。
            # 相当于，网络中所有参数，只有当requires_grad为True的时候，该参数才传给Adam()
            self.optimizer_G = torch.optim.Adam(filter(lambda p: p.requires_grad, itertools.chain(self.netG_A.parameters(), self.netG_B.parameters())), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(filter(lambda p: p.requires_grad, itertools.chain(self.netD_A.parameters(), self.netD_B.parameters())), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers = []
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def select_masks(self, segs_batch):
        """Select instance masks to use"""
        if self.opt.set_order == 'decreasing':
            return self.select_masks_decreasing(segs_batch)
        elif self.opt.set_order == 'random':
            return self.select_masks_random(segs_batch)
        else:
            raise NotImplementedError('Set order name [%s] is not recognized' % self.opt.set_order)

    def select_masks_decreasing(self, segs_batch):
        """Select masks in decreasing order"""	# 猜测是选取前四张大的seg，大的意思是：该seg像素点上的value，总平均值最大
        ret = list()
        for segs in segs_batch:					# 错误理解：
                                                # 	照理说segs_batch类似torch.Size([20, 200, 200])，segs类似torch.Size([1, 200, 200])，
                                                # 	此时下方mean类似torch.Size([1])，那么将无法计算mean.topk(4),因为根本不够4个！
                                                # 实际debug:
                                                # 	查看segs_batch的size是torch.Size([1, 20, 200, 200])，
                                                # 	则segs的size是torch.Size([20, 200, 200])

            mean = segs.mean(-1).mean(-1)		# mean的size是torch.Size([20])
                                                # 这里做了两次mean处理，都是在最后一维进行处理，
                                                # ps. torch.mean():
                                                # 	具体可看/Users/chenqy/PycharmProjects/instagan/data/seg_understanding.py
                                                # 	或者看/instagan/models/torchmean_understanding.py

            m, i = mean.topk(self.opt.ins_max)	# m是：tensor([-0.7352, -0.7675, -1.0000, -1.0000])，大小是torch.Size([4])。
                                                # i是tensor([0, 1, 5, 3])，大小是torch.Size([4]),i可能表示前四个大的seg的索引
                                                # '--ins_max', type=int, default=4, help='maximum number of instances to forward'

            ret.append(segs[i, :, :])			# ret是list，其中每个元素shape是torch.Size([4, 200, 200])

        return torch.stack(ret)					# torch.stack表示在新的dim上concatenate。
                                                # 返回的是torch.Size([1, 4, 200, 200])

    def select_masks_random(self, segs_batch):
        """Select masks in random order"""
        ret = list()
        for segs in segs_batch:
            mean = (segs + 1).mean(-1).mean(-1)															# torch.Size([20])
            m, i = mean.topk(self.opt.ins_max)
            num = min(len(mean.nonzero()), self.opt.ins_max)											# num = {int}2
            reorder = np.concatenate((np.random.permutation(num), np.arange(num, self.opt.ins_max)))	# reorder = {ndarry}[0 1 2 3]
            ret.append(segs[i[reorder], :, :])															# ret是list，其中每个元素shape是torch.Size([4, 200, 200])
        return torch.stack(ret)

    def merge_masks(self, segs):
        """Merge masks (B, N, W, H) -> (B, 1, W, H)"""
        ret = torch.sum((segs + 1)/2, dim=1, keepdim=True)  				# (B, 1, W, H)
        return ret.clamp(max=1, min=0) * 2 - 1

    def get_weight_for_ctx(self, x, y):
        """Get weight for context preserving loss"""
        z = self.merge_masks(torch.cat([x, y], dim=1))
        return (1 - z) / 2  # [-1,1] -> [1,0]

    def weighted_L1_loss(self, src, tgt, weight):
        """L1 loss with given weight (used for context preserving loss)"""
        return torch.mean(weight * torch.abs(src - tgt))

    def get_weight_for_cx(self, x, y):
        """Get weight for context preserving loss"""
        z = self.merge_masks(torch.cat([x, y], dim=1))
        return (1 - z) / 2  # [-1,1] -> [1,0]

    def multiply_cx(self, src, weight):
        """L1 loss with given weight (used for context preserving loss)"""
        return torch.mean(weight * torch.abs(src))

    def split(self, x):
        """Split data into image and mask (only assume 3-channel image)"""
        return x[:, :3, :, :], x[:, 3:, :, :]								# 前三通道是image的，剩余通道是mask的

    # input是数据集实例（类UnalignedSegDataset的实例）
    def set_input(self, input):
        AtoB = self.opt.direction == 'AtoB'
                                                                            # input is the datasets, we use input[idx]to get the item.
                                                                            # eg.input['A'] or input['B'] or input['A_segs'] or input['B_segs']
                                                                            # refer to the "data/unaligned_seg_dataset.py' and see the get_item return the map data
        self.real_A_img = input['A' if AtoB else 'B'].to(self.device)		# self.real_A_img的shape是torch.Size([1, 3, 200, 200])，一张原图，3通道
        self.real_B_img = input['B' if AtoB else 'A'].to(self.device)

        real_A_segs = input['A_segs' if AtoB else 'B_segs']					# real_A_segs是domainA（当AtoB时）中的一张图对应的多张segs，所有segs拼接使用cat函数
        real_B_segs = input['B_segs' if AtoB else 'A_segs']

        self.real_A_segs = self.select_masks(real_A_segs).to(self.device)	# self.real_A_segs的shape是torch.Size([1, 4, 200, 200]),四张seg
        self.real_B_segs = self.select_masks(real_B_segs).to(self.device)

        self.real_A = torch.cat([self.real_A_img, self.real_A_segs], dim=1)	# self.real_A的shape是torch.Size([1, 7, 200, 200])，融合了一张原图和四张seg
        self.real_B = torch.cat([self.real_B_img, self.real_B_segs], dim=1)

        self.real_A_seg = self.merge_masks(self.real_A_segs)  				# merged mask，Merge masks (B, N, W, H) -> (B, 1, W, H)# self.real_A_seg的shape是torch.Size([1, 1, 200, 200])，相当于将其压缩，将7压缩为1
        self.real_B_seg = self.merge_masks(self.real_B_segs)

        self.image_paths = input['A_paths' if AtoB else 'B_paths']			# A_paths是一个list，但是其长度为1，值为'./datasets/shp2gir_coco/trainA/788.png'

    def forward(self, idx=0):
        N = self.opt.ins_per												# '--ins_per', type=int, default=2, help='number of instances to forward, for one pass')	# 一次迭代中，使用到的instance的数目

        self.real_A_seg_sng = self.real_A_segs[:, N*idx:N*(idx+1), :, :]  	# ith mask,似乎取第i批mask，一批有ins_iter张（2张）。sng应该表示single的意思。
                                                                            # self.real_A_segs的shape是torch.Size([1, 4, 200, 200]),四张seg
        self.real_B_seg_sng = self.real_B_segs[:, N*idx:N*(idx+1), :, :]  	# ith mask
        empty = -torch.ones(self.real_A_seg_sng.size()).to(self.device)  	# empty image

        self.forward_A = (self.real_A_seg_sng + 1).sum() > 0  				# check if there are remaining instances
                                                                            # 当forward_A=1时，才前馈并进反向传播
                                                                            # 因为在read_segs()中若seg不存在，则每个像素设置为-1。所以这里(self.real_A_seg_sng + 1)？
        self.forward_B = (self.real_B_seg_sng + 1).sum() > 0  				# check if there are remaining instances

        # forward A
        if self.forward_A:
            self.real_A_sng = torch.cat([self.real_A_img_sng, self.real_A_seg_sng], dim=1)
            self.fake_B_sng = self.netG_A(self.real_A_sng)							# (原图image和掩码)即(self.real_A_sng)作为一个整体输入到生成器
            self.rec_A_sng = self.netG_B(self.fake_B_sng)							# 生成的假的domain B的图（self.fake_B_sng），再输入到G_B进行reconstruc

            self.fake_B_img_sng, self.fake_B_seg_sng = self.split(self.fake_B_sng)	# 生成的假的domain B的图:split分为domainB的假的img和假的seg。
                                                                                    #  暂定self.fake_B_img_sng用于计算IS（inception score）,作为inception网络的输入
            self.rec_A_img_sng, self.rec_A_seg_sng = self.split(self.rec_A_sng)		# reconstruct的domainA的图：split分为domainA的重构的img和假的seg。
                                                                                    #  暂定self.rec_A_img_sng用于计算IS（inception score）
            fake_B_seg_list = self.fake_B_seg_list + [self.fake_B_seg_sng]  		# not detach
            for i in range(self.ins_iter - idx - 1):								# 总共需要增加几个seg？
                fake_B_seg_list.append(empty)										# 为了使fake_B_seg_mul的大小与相同，所以增加一部分seg（这部分seg的每个像素为-1）

            self.fake_B_seg_mul = torch.cat(fake_B_seg_list, dim=1)							# 为什么要计算fake_B_seg_mul，似乎是将多个假的domainB的seg通过torch.cat连接起来
            self.fake_B_mul = torch.cat([self.fake_B_img_sng, self.fake_B_seg_mul], dim=1)	# self.fake_B_mul是假的domainB的结果，用于计算loss

        # forward B
        if self.forward_B:
            self.real_B_sng = torch.cat([self.real_B_img_sng, self.real_B_seg_sng], dim=1)
            self.fake_A_sng = self.netG_B(self.real_B_sng)
            self.rec_B_sng = self.netG_A(self.fake_A_sng)

            self.fake_A_img_sng, self.fake_A_seg_sng = self.split(self.fake_A_sng)																							# 暂定self.fake_A_img_sng用于计算IS（inception score）
            self.rec_B_img_sng, self.rec_B_seg_sng = self.split(self.rec_B_sng)																								# 暂定self.rec_B_img_sng用于计算IS（inception score）
            fake_A_seg_list = self.fake_A_seg_list + [self.fake_A_seg_sng]  # not detach
            for i in range(self.ins_iter - idx - 1):
                fake_A_seg_list.append(empty)

            self.fake_A_seg_mul = torch.cat(fake_A_seg_list, dim=1)
            self.fake_A_mul = torch.cat([self.fake_A_img_sng, self.fake_A_seg_mul], dim=1)

    def test(self):															# 用于test.py
        # init setting														# 与optimize_parameters()相同的初始化
        self.real_A_img_sng = self.real_A_img								# self.real_A_img的shape是torch.Size([1, 3, 200, 200])，一张原图，3通道
        self.real_B_img_sng = self.real_B_img
        self.fake_A_seg_list = list()
        self.fake_B_seg_list = list()
        self.rec_A_seg_list = list()
        self.rec_B_seg_list = list()

        # sequential mini-batch translation
        for i in range(self.ins_iter):
            # forward
            with torch.no_grad():  											# no grad,注意！test的时候没有更新参数，所以forward的时候设置：no grad
                self.forward(i)

            # update setting for next iteration
            self.real_A_img_sng = self.fake_B_img_sng.detach()
            self.real_B_img_sng = self.fake_A_img_sng.detach()
            self.fake_A_seg_list.append(self.fake_A_seg_sng.detach())
            self.fake_B_seg_list.append(self.fake_B_seg_sng.detach())
            self.rec_A_seg_list.append(self.rec_A_seg_sng.detach())
            self.rec_B_seg_list.append(self.rec_B_seg_sng.detach())

            # save visuals
            if i == 0:  # first
                self.rec_A_img = self.rec_A_img_sng
                self.rec_B_img = self.rec_B_img_sng
            if i == self.ins_iter - 1:  # last
                self.fake_A_img = self.fake_A_img_sng
                self.fake_B_img = self.fake_B_img_sng
                self.fake_A_seg = self.merge_masks(self.fake_A_seg_mul)
                self.fake_B_seg = self.merge_masks(self.fake_B_seg_mul)
                self.rec_A_seg = self.merge_masks(torch.cat(self.rec_A_seg_list, dim=1))
                self.rec_B_seg = self.merge_masks(torch.cat(self.rec_B_seg_list, dim=1))

    def backward_G(self):													# 计算生成器的总loss并反向传播
        lambda_A = self.opt.lambda_A										# 用于backward A
        lambda_B = self.opt.lambda_B										# 用于backward B
        lambda_idt = self.opt.lambda_idt									# 用于loss_idt_A和loss_idt_B
        lambda_ctx = self.opt.lambda_ctx

        # backward A
        if self.forward_A:
            self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B_mul), True)														# self.fake_B_mul用于计算loss，假的B的结果经过判别器，其结果（self.netD_A(self.fake_B_mul)）与希望的结果（True）之间的loss
            self.loss_cyc_A = self.criterionCyc(self.rec_A_sng, self.real_A_sng) * lambda_A
            self.loss_idt_B = self.criterionIdt(self.netG_B(self.real_A_sng), self.real_A_sng.detach()) * lambda_A * lambda_idt			# self.real_A_sng.detach()
            weight_A = self.get_weight_for_ctx(self.real_A_seg_sng, self.fake_B_seg_sng)
            self.loss_ctx_A = self.weighted_L1_loss(self.real_A_img_sng, self.fake_B_img_sng, weight=weight_A) * lambda_A * lambda_ctx
            layers = {"conv_1_1": 1.0,"conv_3_2": 1.0}
            I = torch.rand(1, 3, 128, 128).cuda()
            T = torch.randn(1, 3, 128, 128).cuda()
            I = self.fake_B_mul  # 生成的B域的图
            T = self.real_B_sng  # 目标域B的真实图
            I = self.fake_B_img_sng
            T = self.real_B_img_sng
            I_multiply = self.fake_B_seg_mul * I
            T_multiply = self.real_B_seg_sng * T

            x_image_numpy = util.tensor2im(I)
            x_image_pil = Image.fromarray(x_image_numpy)
            # x_image_pil.show()
            y_image_numpy = util.tensor2im(T)
            y_image_pil = Image.fromarray(y_image_numpy)
            # y_image_pil.show()

            x_image_numpy = util.tensor2im(self.fake_B_seg_mul)
            x_image_pil = Image.fromarray(x_image_numpy)
            # x_image_pil.show()
            y_image_numpy = util.tensor2im(self.real_B_seg_sng)
            y_image_pil = Image.fromarray(y_image_numpy)
            # y_image_pil.show()

            x_image_numpy = util.tensor2im(I_multiply)
            x_image_pil = Image.fromarray(x_image_numpy)
            # x_image_pil.show()
            y_image_numpy = util.tensor2im(T_multiply)
            y_image_pil = Image.fromarray(y_image_numpy)
            # y_image_pil.show()

            contex_loss = Contextual_Loss(layers, max_1d_size=64).cuda()
            # print('cxloss_A', contex_loss(I_multiply, T_multiply))
            self.loss_cx_A = contex_loss(I_multiply, T_multiply)[0]
        else:
            self.loss_G_A = 0
            self.loss_cyc_A = 0
            self.loss_idt_B = 0
            self.loss_ctx_A = 0

        # backward B
        if self.forward_B:
            self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A_mul), True)
            self.loss_cyc_B = self.criterionCyc(self.rec_B_sng, self.real_B_sng) * lambda_B
            self.loss_idt_A = self.criterionIdt(self.netG_A(self.real_B_sng), self.real_B_sng.detach()) * lambda_B * lambda_idt
            weight_B = self.get_weight_for_ctx(self.real_B_seg_sng, self.fake_A_seg_sng)
            self.loss_ctx_B = self.weighted_L1_loss(self.real_B_img_sng, self.fake_A_img_sng, weight=weight_B) * lambda_B * lambda_ctx
            '''layers = {"conv_1_1": 1.0, "conv_3_2": 1.0}
            I = torch.rand(1, 3, 128, 128).cuda()
            T = torch.randn(1, 3, 128, 128).cuda()
            I = self.fake_A_mul  # 生成的B域的图
            T = self.real_A_sng  # 目标域B的真实图
            contex_loss = Contextual_Loss(layers, max_1d_size=64).cuda()
            print('cxloss_B', contex_loss(I, T))'''
            layers = {"conv_1_1": 1.0, "conv_3_2": 1.0}
            I = torch.rand(1, 3, 128, 128).cuda()
            T = torch.randn(1, 3, 128, 128).cuda()
            I = self.fake_A_mul  # 生成的B域的图
            T = self.real_A_sng  # 目标域B的真实图
            I = self.fake_A_img_sng
            T = self.real_A_img_sng
            I_multiply = self.fake_A_seg_mul * I
            T_multiply = self.real_A_seg_sng * T

            x_image_numpy = util.tensor2im(I)
            x_image_pil = Image.fromarray(x_image_numpy)
            # x_image_pil.show()
            y_image_numpy = util.tensor2im(T)
            y_image_pil = Image.fromarray(y_image_numpy)
            # y_image_pil.show()

            x_image_numpy = util.tensor2im(self.fake_B_seg_mul)
            x_image_pil = Image.fromarray(x_image_numpy)
            # x_image_pil.show()
            y_image_numpy = util.tensor2im(self.real_B_seg_sng)
            y_image_pil = Image.fromarray(y_image_numpy)
            # y_image_pil.show()

            x_image_numpy = util.tensor2im(I_multiply)
            x_image_pil = Image.fromarray(x_image_numpy)
            # x_image_pil.show()
            y_image_numpy = util.tensor2im(T_multiply)
            y_image_pil = Image.fromarray(y_image_numpy)
            # y_image_pil.show()

            contex_loss = Contextual_Loss(layers, max_1d_size=64).cuda()
            # print('cxloss_B', contex_loss(I_multiply, T_multiply))
            self.loss_cx_B = contex_loss(I_multiply, T_multiply)[0]
        else:
            self.loss_G_B = 0
            self.loss_cyc_B = 0
            self.loss_idt_A = 0
            self.loss_ctx_B = 0

        # combined loss
        # self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cyc_A + self.loss_cyc_B + self.loss_idt_A + self.loss_idt_B + self.loss_ctx_A + self.loss_ctx_B
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cyc_A + self.loss_cyc_B + self.loss_idt_A + self.loss_idt_B + self.loss_ctx_A + self.loss_ctx_B + self.loss_cx_A + self.loss_cx_B
        self.loss_G.backward()	# 生成器A和生成器B的各种loss为总G的loss，反向传播

    def backward_D_basic(self, netD, real, fake):
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        # backward
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        fake_B = self.fake_B_pool.query(self.fake_B_mul)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self):
        fake_A = self.fake_A_pool.query(self.fake_A_mul)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

    def optimize_parameters(self):											# 用于train.py,和test()很像
        # init setting														# 与test()相同的初始化
        self.real_A_img_sng = self.real_A_img								# self.real_A_img的shape是torch.Size([1, 3, 200, 200])，一张原图，3通道
        self.real_B_img_sng = self.real_B_img
        self.fake_A_seg_list = list()
        self.fake_B_seg_list = list()
        self.rec_A_seg_list = list()
        self.rec_B_seg_list = list()

        # sequential mini-batch translation
        for i in range(self.ins_iter):
            # forward
            self.forward(i)

            # G_A and G_B													# 比test多出的部分
            if self.forward_A or self.forward_B:
                self.set_requires_grad([self.netD_A, self.netD_B], False)	# 为什么设置判别器A和判别器B的参数不需要更新？
                self.optimizer_G.zero_grad()
                self.backward_G()											# 生成器的loss的反向传播
                self.optimizer_G.step()										# 更新参数

            # D_A and D_B													# 比test多出的部分
            if self.forward_A or self.forward_B:
                self.set_requires_grad([self.netD_A, self.netD_B], True)	# 设置判别器的参数需要更新
                self.optimizer_D.zero_grad()
                if self.forward_A:
                    self.backward_D_A()										# 判别器A的loss的反向传播，为什么判别器要分开反向传播？
                if self.forward_B:
                    self.backward_D_B()										# 判别器B的loss的反向传播
                self.optimizer_D.step()										# 更新参数

            # update setting for next iteration
            self.real_A_img_sng = self.fake_B_img_sng.detach()
            self.real_B_img_sng = self.fake_A_img_sng.detach()
            self.fake_A_seg_list.append(self.fake_A_seg_sng.detach())
            self.fake_B_seg_list.append(self.fake_B_seg_sng.detach())
            self.rec_A_seg_list.append(self.rec_A_seg_sng.detach())
            self.rec_B_seg_list.append(self.rec_B_seg_sng.detach())

            # save visuals
            if i == 0:  # first
                self.rec_A_img = self.rec_A_img_sng
                self.rec_B_img = self.rec_B_img_sng
            if i == self.ins_iter - 1:  # last
                self.fake_A_img = self.fake_A_img_sng
                self.fake_B_img = self.fake_B_img_sng
                self.fake_A_seg = self.merge_masks(self.fake_A_seg_mul)
                self.fake_B_seg = self.merge_masks(self.fake_B_seg_mul)
                self.rec_A_seg = self.merge_masks(torch.cat(self.rec_A_seg_list, dim=1))
                self.rec_B_seg = self.merge_masks(torch.cat(self.rec_B_seg_list, dim=1))
