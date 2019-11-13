import torch
import torch.nn as nn
import torchsnooper
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import torch.nn.functional as F
from collections import OrderedDict

###############################################################################
# Helper Functions
###############################################################################


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer   # 返回偏函数，功能类似norm层


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type, gain=init_gain)
    return net


def define_G(input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[], growth_rate=32, block_config=(6, 12, 24, 16),num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000, memory_efficient=False):
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netG == 'basic':
        # net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
        net = DensenetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9, growth_rate=growth_rate, block_config=block_config,num_init_features=num_init_features, bn_size=bn_size, drop_rate=drop_rate, num_classes=num_classes, memory_efficient=memory_efficient)
    elif netG == 'set':
        # net = ResnetSetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
        net = DensenetSetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9, growth_rate=growth_rate, block_config=block_config,num_init_features=num_init_features, bn_size=bn_size, drop_rate=drop_rate, num_classes=num_classes, memory_efficient=memory_efficient)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    return init_net(net, init_type, init_gain, gpu_ids)


def define_D(input_nc, ndf, netD, n_layers_D=3, norm='batch', use_sigmoid=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'basic':
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    elif netD == 'set':
        net = NLayerSetDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % net)
    return init_net(net, init_type, init_gain, gpu_ids)


##############################################################################
# Classes
##############################################################################


# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
# 计算ganloss，已知常规形式：loss（output，target）。这里output就是以下的self.input，target就是target_real_label.
# 注意loss（output，target），在此处，output和target都是tensor
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    # 将target_is_real（形式很有可能是boolean，比如True或False）转成tensor，且大小和input一致
    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)


# Define spectral normalization layer
# Code from Christian Cosgrove's repository
# https://github.com/christiancosgrove/pytorch-spectral-normalization-gan/blob/master/spectral_normalization.py
def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)

class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height, -1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height, -1).data, v.data))

        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = nn.Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = nn.Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = nn.Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)


# Defines the generator that consists of Resnet blocks between a few
# downsampling/upsampling operations.
# Code and idea originally from Justin Johnson's architecture.
# https://github.com/jcjohnson/fast-neural-style/
class ResnetGenerator(nn.Module):   # 使用resnet作为生成器的backbone net
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=9, padding_type='reflect'):
        assert(n_blocks >= 0)   # 默认9个resnet block
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        if type(norm_layer) == functools.partial:           # functools.partial：是一种类型。输入：type(functools.partial),输出type. 判断norm_layer的类型是不是偏函数
            use_bias = norm_layer.func == nn.InstanceNorm2d # 是偏函数
        else:
            use_bias = norm_layer == nn.InstanceNorm2d      # 不是偏函数

        model = [nn.ReflectionPad2d(3),     # [1,3,206,206]
                                                            #  加padding，padding的值，为对称关系，refer：https://pytorch.org/docs/stable/nn.html#reflectionpad2d
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                           bias=use_bias),  # [1,64,200,200]
                 norm_layer(ngf),           # [1,64,200,200]instancenorm的输出大小与输入相同
                 nn.ReLU(True)]             # [1,64,200,200]与输入相同

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i # 计算2的i次方
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):     # i = 0,1．循环两次
            mult = 2**(n_downsampling - i)  # mult = 4
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        # model如下：
        # <class 'list'>: [ReflectionPad2d((3, 3, 3, 3)), Conv2d(3, 64, kernel_size=(7, 7), stride=(1, 1)),
        #                 InstanceNorm2d(64, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False),
        #                 ReLU(inplace), Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
        #                 InstanceNorm2d(128, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False),
        #                 ReLU(inplace), Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
        #                 InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False),
        #                 ReLU(inplace),
        # ResnetBlock(
        #      (conv_block): Sequential(
        # (0): ReflectionPad2d((1, 1, 1, 1))
        # (1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1))
        # (2): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        # (3): ReLU(inplace)
        # (4): ReflectionPad2d((1, 1, 1, 1))
        # (5): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1))
        # (6): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        # )
        # ), ResnetBlock(
        #     (conv_block): Sequential(
        # (0): ReflectionPad2d((1, 1, 1, 1))
        # (1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1))
        # (2): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        # (3): ReLU(inplace)
        # (4): ReflectionPad2d((1, 1, 1, 1))
        # (5): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1))
        # (6): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        # )
        # ), ResnetBlock(
        #     (conv_block): Sequential(
        # (0): ReflectionPad2d((1, 1, 1, 1))
        # (1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1))
        # (2): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        # (3): ReLU(inplace)
        # (4): ReflectionPad2d((1, 1, 1, 1))
        # (5): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1))
        # (6): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        # )
        # ), ResnetBlock(
        #     (conv_block): Sequential(
        # (0): ReflectionPad2d((1, 1, 1, 1))
        # (1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1))
        # (2): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        # (3): ReLU(inplace)
        # (4): ReflectionPad2d((1, 1, 1, 1))
        # (5): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1))
        # (6): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        # )
        # ), ResnetBlock(
        #     (conv_block): Sequential(
        #     (0): ReflectionPad2d((1, 1, 1, 1))
        # (1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1))
        # (2): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        # (3): ReLU(inplace)
        # (4): ReflectionPad2d((1, 1, 1, 1))
        # (5): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1))
        # (6): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        # )
        # ), ResnetBlock(
        #     (conv_block): Sequential(
        # (0): ReflectionPad2d((1, 1, 1, 1))
        # (1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1))
        # (2): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        # (3): ReLU(inplace)
        # (4): ReflectionPad2d((1, 1, 1, 1))
        # (5): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1))
        # (6): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        # )
        # ), ResnetBlock(
        #     (conv_block): Sequential(
        # (0): ReflectionPad2d((1, 1, 1, 1))
        # (1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1))
        # (2): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        # (3): ReLU(inplace)
        # (4): ReflectionPad2d((1, 1, 1, 1))
        # (5): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1))
        # (6): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        # )
        # ), ResnetBlock(
        #     (conv_block): Sequential(
        # (0): ReflectionPad2d((1, 1, 1, 1))
        # (1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1))
        # (2): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        # (3): ReLU(inplace)
        # (4): ReflectionPad2d((1, 1, 1, 1))
        # (5): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1))
        # (6): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        # )
        # ), ResnetBlock(
        #     (conv_block): Sequential(
        # (0): ReflectionPad2d((1, 1, 1, 1))
        # (1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1))
        # (2): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        # (3): ReLU(inplace)
        # (4): ReflectionPad2d((1, 1, 1, 1))
        # (5): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1))
        # (6): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        # )
        # ),
        # ConvTranspose2d(256, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1)),
        # InstanceNorm2d(128, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False),
        # ReLU(inplace),
        # ConvTranspose2d(128, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1)),
        # InstanceNorm2d(64, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False), ReLU(inplace),
        # ReflectionPad2d((3, 3, 3, 3)),
        # Conv2d(64, 3, kernel_size=(7, 7), stride=(1, 1)),
        # Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


# ResNet generator for "set" of instance attributes
# See https://openreview.net/forum?id=ryxwJhC9YX for details
class ResnetSetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        assert (n_blocks >= 0)
        super(ResnetSetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        n_downsampling = 2
        self.encoder_img = self.get_encoder(input_nc, n_downsampling, ngf, norm_layer, use_dropout, n_blocks, padding_type, use_bias)
        self.encoder_seg = self.get_encoder(1, n_downsampling, ngf, norm_layer, use_dropout, n_blocks, padding_type, use_bias)
        self.decoder_img = self.get_decoder(output_nc, n_downsampling, 2 * ngf, norm_layer, use_bias)   # 2*ngf,此时新的ngf变成128
        self.decoder_seg = self.get_decoder(1, n_downsampling, 3 * ngf, norm_layer, use_bias)           # 3*ngf,因为输入的channel大小是3倍

    def get_encoder(self, input_nc, n_downsampling, ngf, norm_layer, use_dropout, n_blocks, padding_type, use_bias):
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        for i in range(n_downsampling):
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        # encoder的model如下：
        # <class 'list'>: [ReflectionPad2d((3, 3, 3, 3)), Conv2d(3, 64, kernel_size=(7, 7), stride=(1, 1)),
        #                 InstanceNorm2d(64, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False),
        #                 ReLU(inplace), Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
        #                 InstanceNorm2d(128, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False),
        #                 ReLU(inplace), Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
        #                 InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False),
        #                 ReLU(inplace),
        # ResnetBlock(
        #         (conv_block): Sequential(
        # (0): ReflectionPad2d((1, 1, 1, 1))
        # (1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1))
        # (2): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        # (3): ReLU(inplace)
        # (4): ReflectionPad2d((1, 1, 1, 1))
        # (5): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1))
        # (6): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        # )
        # ), ResnetBlock(
        #     (conv_block): Sequential(
        # (0): ReflectionPad2d((1, 1, 1, 1))
        # (1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1))
        # (2): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        # (3): ReLU(inplace)
        # (4): ReflectionPad2d((1, 1, 1, 1))
        # (5): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1))
        # (6): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        # )
        # ), ResnetBlock(
        #     (conv_block): Sequential(
        # (0): ReflectionPad2d((1, 1, 1, 1))
        # (1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1))
        # (2): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        # (3): ReLU(inplace)
        # (4): ReflectionPad2d((1, 1, 1, 1))
        # (5): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1))
        # (6): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        # )
        # ), ResnetBlock(
        #     (conv_block): Sequential(
        # (0): ReflectionPad2d((1, 1, 1, 1))
        # (1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1))
        # (2): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        # (3): ReLU(inplace)
        # (4): ReflectionPad2d((1, 1, 1, 1))
        # (5): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1))
        # (6): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        # )
        # ), ResnetBlock(
        #     (conv_block): Sequential(
        # (0): ReflectionPad2d((1, 1, 1, 1))
        # (1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1))
        # (2): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        # (3): ReLU(inplace)
        # (4): ReflectionPad2d((1, 1, 1, 1))
        # (5): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1))
        # (6): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        # )
        # ), ResnetBlock(
        #     (conv_block): Sequential(
        # (0): ReflectionPad2d((1, 1, 1, 1))
        # (1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1))
        # (2): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        # (3): ReLU(inplace)
        # (4): ReflectionPad2d((1, 1, 1, 1))
        # (5): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1))
        # (6): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        # )
        # ), ResnetBlock(
        #     (conv_block): Sequential(
        # (0): ReflectionPad2d((1, 1, 1, 1))
        # (1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1))
        # (2): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        # (3): ReLU(inplace)
        # (4): ReflectionPad2d((1, 1, 1, 1))
        # (5): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1))
        # (6): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        # )
        # ), ResnetBlock(
        #     (conv_block): Sequential(
        # (0): ReflectionPad2d((1, 1, 1, 1))
        # (1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1))
        # (2): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        # (3): ReLU(inplace)
        # (4): ReflectionPad2d((1, 1, 1, 1))
        # (5): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1))
        # (6): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        # )
        # ), ResnetBlock(
        #     (conv_block): Sequential(
        # (0): ReflectionPad2d((1, 1, 1, 1))
        # (1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1))
        # (2): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        # (3): ReLU(inplace)
        # (4): ReflectionPad2d((1, 1, 1, 1))
        # (5): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1))
        # (6): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        # )
        # )]

        return nn.Sequential(*model)

    def get_decoder(self, output_nc, n_downsampling, ngf, norm_layer, use_bias):
        model = []
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        # decoder的model如下：
        # <class 'list'>: [
        #     ConvTranspose2d(512, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1)),
        #     InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False),
        #     ReLU(inplace),
        #     ConvTranspose2d(256, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1)),
        #     InstanceNorm2d(128, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False),
        #     ReLU(inplace),
        #     ReflectionPad2d((3, 3, 3, 3)),
        #     Conv2d(128, 3, kernel_size=(7, 7), stride=(1, 1)),
        #     Tanh()]

        return nn.Sequential(*model)

    def forward(self, inp):
        # split data
        img = inp[:, :self.input_nc, :, :]              # (B, CX, W, H) img:torch.Size([1, 3, 200, 200])
        segs = inp[:, self.input_nc:, :, :]             # (B, CA, W, H) segs:torch.Size([1, 2, 200, 200])
        mean = (segs + 1).mean(0).mean(-1).mean(-1)     # mean:torch.Size([2]) mean = {Tensor}tensor([0.0523, 0.0469], device='cuda:0')
        if mean.sum() == 0:
            mean[0] = 1  # forward at least one segmentation

        # run encoder
        enc_img = self.encoder_img(img)                             # enc_img:torch.Size([1, 256, 50, 50]) encoder没有改变ｘ[1,256,50,50]的大小
        enc_segs = list()
        for i in range(segs.size(1)):
            if mean[i] > 0:                                         # skip empty segmentation
                seg = segs[:, i, :, :].unsqueeze(1)                 # seg:torch.Size([1, 1, 200, 200])
                enc_segs.append(self.encoder_seg(seg))              # self.encoder_seg(seg)的结果torch.Size([1, 256, 50, 50]),总共append两次
        enc_segs = torch.cat(enc_segs)
        enc_segs_sum = torch.sum(enc_segs, dim=0, keepdim=True)     # enc_segs_sum:torch.Size([1, 256, 50, 50])
                                                                    #  aggregated set feature

        # run decoder
        feat = torch.cat([enc_img, enc_segs_sum], dim=1)            # feat:torch.Size([1, 512, 50, 50])
        out = [self.decoder_img(feat)]
        idx = 0
        for i in range(segs.size(1)):
            if mean[i] > 0:
                enc_seg = enc_segs[idx].unsqueeze(0)                # (1, ngf, w, h)
                idx += 1  # move to next index
                feat = torch.cat([enc_seg, enc_img, enc_segs_sum], dim=1)
                out += [self.decoder_seg(feat)]
            else:
                out += [segs[:, i, :, :].unsqueeze(1)]              # skip empty segmentation
        return torch.cat(out, dim=1)


# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        # net_G为basic模式时，conv_block如下：
        # <class 'list'>: [ReflectionPad2d((1, 1, 1, 1)),
        #                  Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1)),
        #                  InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False),
        #                  ReLU(inplace),
        #                  ReflectionPad2d((1, 1, 1, 1)),
        #                  Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1)),
        #                  InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)]

        # net_G为set模式时，conv_block如下：
        # <class 'list'>: [ReflectionPad2d((1, 1, 1, 1)),
        #                  Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1)),
        #                 InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False),
        #                 ReLU(inplace),
        #                 ReflectionPad2d((1, 1, 1, 1)),
        #                 Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1)),
        #                 InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)    # x:torch.Size([1, 256, 50, 50])
        return out                      # out:torch.Size([1, 256, 50, 50])


# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetGenerator, self).__init__()

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)

        self.model = unet_block

    def forward(self, input):
        return self.model(input)


# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [
            # Use spectral normalization
            SpectralNorm(nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw)),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                # Use spectral normalization
                SpectralNorm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias)),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            # Use spectral normalization
            SpectralNorm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias)),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        # Use spectral normalization
        sequence += [SpectralNorm(nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw))]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)


# PatchGAN discriminator for "set" of instance attributes
# See https://openreview.net/forum?id=ryxwJhC9YX for details
class NLayerSetDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(NLayerSetDiscriminator, self).__init__()
        self.input_nc = input_nc
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        self.feature_img = self.get_feature_extractor(input_nc, ndf, n_layers, kw, padw, norm_layer, use_bias)
        self.feature_seg = self.get_feature_extractor(1, ndf, n_layers, kw, padw, norm_layer, use_bias)
        self.classifier = self.get_classifier(2 * ndf, n_layers, kw, padw, norm_layer, use_sigmoid)  # 2*ndf

    def get_feature_extractor(self, input_nc, ndf, n_layers, kw, padw, norm_layer, use_bias):
        model = [
            # Use spectral normalization
            SpectralNorm(nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw)),
            nn.LeakyReLU(0.2, True)
        ]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            model += [
                # Use spectral normalization
                SpectralNorm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias)),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]
        return nn.Sequential(*model)

    def get_classifier(self, ndf, n_layers, kw, padw, norm_layer, use_sigmoid):
        nf_mult_prev = min(2 ** (n_layers-1), 8)
        nf_mult = min(2 ** n_layers, 8)
        model = [
            # Use spectral normalization
            SpectralNorm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw)),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]
        # Use spectral normalization
        model += [SpectralNorm(nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw))]
        if use_sigmoid:
            model += [nn.Sigmoid()]
        return nn.Sequential(*model)

    def forward(self, inp):
        # split data
        img = inp[:, :self.input_nc, :, :]  # (B, CX, W, H)
        segs = inp[:, self.input_nc:, :, :]  # (B, CA, W, H)
        mean = (segs + 1).mean(0).mean(-1).mean(-1)
        if mean.sum() == 0:
            mean[0] = 1  # forward at least one segmentation

        # run feature extractor
        feat_img = self.feature_img(img)
        feat_segs = list()
        for i in range(segs.size(1)):
            if mean[i] > 0:  # skip empty segmentation
                seg = segs[:, i, :, :].unsqueeze(1)
                feat_segs.append(self.feature_seg(seg))
        feat_segs_sum = torch.sum(torch.stack(feat_segs), dim=0)  # aggregated set feature

        # run classifier
        feat = torch.cat([feat_img, feat_segs_sum], dim=1)
        out = self.classifier(feat)
        return out


class PixelDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        if use_sigmoid:
            self.net.append(nn.Sigmoid())

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        return self.net(input)


# # Densenet Block的子模块_DenseLayer
# class _DenseLayer(nn.Sequential):
#     def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
#         super(_DenseLayer, self).__init__()
#         self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
#         self.add_module('relu1', nn.ReLU(inplace=True)),
#         self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
#                                            growth_rate, kernel_size=1, stride=1,
#                                            bias=False)),
#         self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
#         self.add_module('relu2', nn.ReLU(inplace=True)),
#         self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
#                                            kernel_size=3, stride=1, padding=1,
#                                            bias=False)),
#         self.drop_rate = drop_rate
#
#     def forward(self, x):
#         new_features = super(_DenseLayer, self).forward(x)  # x:torch.Size([1, 256, 50, 50]) new_features:torch.Size([1, 32, 50, 50]) next:x:torch.Size([1, 288, 50, 50])
#         if self.drop_rate > 0:
#             new_features = F.dropout(new_features, p=self.drop_rate,
#                                      training=self.training)
#         return torch.cat([x, new_features], 1)


# Densenet Block的子模块_DenseLayer
class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1,
                                           bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1,
                                           bias=False)),
        self.drop_rate = drop_rate

        self.norm1 = nn.BatchNorm2d(num_input_features)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False)
        self.norm2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        norm1_out = self.norm1(x)   # seg输入出现错误，seg作为ｘ，大小为torch.Size([1, 1, 200, 200])
        relu1_out = self.relu1(norm1_out)
        conv1_out = self.conv1(relu1_out)
        norm2_out = self.norm2(conv1_out)
        relu2_out = self.relu2(norm2_out)
        conv2_out = self.conv2(relu2_out)
        new_features = conv2_out

        # new_features = super(_DenseLayer, self).forward(x)  # x:torch.Size([1, 256, 50, 50]) new_features:torch.Size([1, 32, 50, 50]) next:x:torch.Size([1, 288, 50, 50])
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate,
                                     training=self.training)
        cat_out = torch.cat([x, new_features], 1)
        return torch.cat([x, new_features], 1)

# Densenet Block的子模块_Transition
class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        # self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))    # 不avgpool，使图片大小不变


# Densenet Block的子模块_DenseBlock
class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate,
                                bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


# Define a Densenet block
class DensenetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias,
                 growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000):
        super(DensenetBlock, self).__init__()
        # self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)
        self.dense_block = self.build_dense_block(dim, padding_type, norm_layer, use_dropout, use_bias,
                                                  growth_rate, block_config,
                                                  num_init_features, bn_size, drop_rate, num_classes)



    def build_dense_block(self, dim, padding_type, norm_layer, use_dropout, use_bias,
                          growth_rate=32, block_config=(6, 12, 24, 16),
                          num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000):

        densenetblock = []
        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate,
                                drop_rate=drop_rate)
            # self.features.add_module('denseblock%d' % (i + 1), block)
            densenetblock += block

            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2)
                # self.features.add_module('transition%d' % (i + 1), trans)
                densenetblock += trans
                num_features = num_features // 2

        return nn.Sequential(*densenetblock)    # 必须有＊号

    def forward(self, x):
        out = self.dense_block(x)   # x:torch.Size([1, 256, 50, 50])
        return out


class DensenetGenerator(nn.Module):   # 使用densenet作为生成器的backbone net
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=9, padding_type='reflect', growth_rate=32, block_config=(6, 12, 24, 16),num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000, memory_efficient=False):
        assert(n_blocks >= 0)   # 默认9个resnet block
        super(DensenetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.growth_rate = growth_rate
        self.block_config = block_config
        self.num_init_num_features = num_init_features
        self.bn_size = bn_size
        self.drop_rate = drop_rate
        self.num_classes = num_classes
        self.memory_efficient = memory_efficient
        if type(norm_layer) == functools.partial:           # functools.partial：是一种类型。输入：type(functools.partial),输出type. 判断norm_layer的类型是不是偏函数
            use_bias = norm_layer.func == nn.InstanceNorm2d # 是偏函数
        else:
            use_bias = norm_layer == nn.InstanceNorm2d      # 不是偏函数


        model = [nn.ReflectionPad2d(3),                     # 加padding，padding的值，为对称关系，refer：https://pytorch.org/docs/stable/nn.html#reflectionpad2d
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i # 计算2的i次方
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        # First convolution
        model += nn.Conv2d(3, num_init_features, kernel_size=7, stride=2,
                                padding=3, bias=False)
        model += nn.BatchNorm2d(num_init_features)
        model += nn.ReLU(inplace=True)
        model += nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # self.features = nn.Sequential(OrderedDict([
        #     ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2,
        #                         padding=3, bias=False)),
        #     ('norm0', nn.BatchNorm2d(num_init_features)),
        #     ('relu0', nn.ReLU(inplace=True)),
        #     ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        # ]))

        mult = 2**n_downsampling

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate,
                                drop_rate=drop_rate)
            model += block
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2)
                model += trans
                num_features = num_features // 2





            # model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
            # model += [DensenetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias, growth_rate=32, block_config=(6, 12, 24, 16),num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000)]

        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

        # 打印该模型
        # from torchsummary import summary
        # H = 200                                             # 暂定为200,这里H等于输入的self.real_img_sng的H
        # W = 200                                             # 暂定为200,这里H等于输入的self.real_img_sng的H
        # summary(self.model, input_size=(input_nc, H, W))

    def forward(self, input):
        return self.model(input)

# @torchsnooper.snoop()
class DensenetSetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect', growth_rate=32, block_config=(6, 12, 24, 16),num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000, memory_efficient=False):
        assert (n_blocks >= 0)
        super(DensenetSetGenerator, self).__init__()

        if torch.cuda.is_available():
            self.dtype  = torch.cuda.FloatTensor
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.growth_rate = growth_rate
        self.block_config = block_config
        self.num_init_num_features = num_init_features
        self.bn_size = bn_size
        self.drop_rate = drop_rate
        self.num_classes = num_classes
        self.memory_efficient = memory_efficient
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        n_downsampling = 2
        self.encoder_img = self.get_encoder(input_nc, n_downsampling, ngf, norm_layer, use_dropout, n_blocks, padding_type, use_bias, growth_rate, block_config, num_init_features,bn_size, drop_rate, num_classes)
        self.encoder_seg = self.get_encoder(1, n_downsampling, ngf, norm_layer, use_dropout, n_blocks, padding_type, use_bias, growth_rate, block_config, num_init_features,bn_size, drop_rate, num_classes)

        self.decoder_img = self.get_decoder(output_nc, n_downsampling, 2 * ngf, norm_layer, use_bias,growth_rate, block_config, num_init_features, bn_size, drop_rate, num_classes)  # 2*ngf
        self.decoder_seg = self.get_decoder(1, n_downsampling, 3 * ngf, norm_layer, use_bias,growth_rate, block_config, num_init_features, bn_size, drop_rate, num_classes)  # 3*ngf

        self.conv0 = nn.Conv2d(input_nc, num_init_features, kernel_size=7, stride=2, padding=3,bias=False)  # 这里的num_init_features和resnetGenerator中的ngf是一个意思
        self.conv0_seg = nn.Conv2d(1, num_init_features, kernel_size=7, stride=2, padding=3,
                               bias=False)  # 这里的num_init_features和resnetGenerator中的ngf是一个意思 # TODO
        self.norm0 = nn.BatchNorm2d(num_init_features)
        self.relu0 = nn.ReLU(inplace=True)
        self.maxpool0 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.dense_ngf = self.get_dense_ngf(growth_rate, block_config, num_init_features, bn_size, drop_rate, num_classes)

        self.decoder_convTran_dict = OrderedDict()
        self.decoder_norm_dict = OrderedDict()
        self.decoder_relu_dict = OrderedDict()

        self.decoder_convTran_dict_seg = OrderedDict()
        self.decoder_norm_dict_seg = OrderedDict()
        self.decoder_relu_dict_seg = OrderedDict()


        self.get_dense_decoder = self.set_dense_decoder(output_nc, n_downsampling, ngf, norm_layer, use_bias, growth_rate, block_config, num_init_features, bn_size, drop_rate, num_classes, num_times=2)
        self.get_dense_decoder_seg =  self.set_dense_decoder_seg(output_nc, n_downsampling, ngf, norm_layer, use_bias, growth_rate, block_config, num_init_features, bn_size, drop_rate, num_classes, num_times=3)


        self.reflec = nn.ReflectionPad2d(3)
        self.convlast = nn.Conv2d(self.last_dense_ngf, output_nc, kernel_size=7, padding=0)
        self.convlast_seg = nn.Conv2d(self.last_dense_ngf, 1, kernel_size=7, padding=0)
        self.tanh = nn.Tanh()



    def get_dense_ngf(self, growth_rate, block_config, num_init_features, bn_size, drop_rate, num_classes):
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):   # 其实每个block后，通道数都变为(num_init_features + num_layers * growth_rate)
            num_features = (num_init_features + num_layers * growth_rate)

        dense_ngf = num_features
        return dense_ngf

    def set_dense_decoder(self, output_nc, n_downsampling, ngf, norm_layer, use_bias, growth_rate, block_config, num_init_features, bn_size, drop_rate, num_classes, num_times):
        dense_ngf = self.get_dense_ngf(growth_rate, block_config, num_init_features, bn_size, drop_rate, num_classes)
        dense_downsampling = 2  # 因为缩小了5倍
        for i in range(dense_downsampling):
            self.decoder_convTran_dict['decoder_convTran%d' % (i + 1)] = nn.ConvTranspose2d(int(dense_ngf * num_times),int(dense_ngf),kernel_size=3, stride=2,padding=1, output_padding=1,bias=use_bias).type(self.dtype)
            self.decoder_norm_dict['decoder_norm%d' % (i + 1)] = norm_layer(int(dense_ngf)).type(self.dtype)
            self.decoder_relu_dict['decoder_relu%d' % (i + 1)] = nn.ReLU(True).type(self.dtype)
            if i!= (dense_downsampling-1):
                dense_ngf = dense_ngf // num_times
        self.last_dense_ngf = dense_ngf

    def set_dense_decoder_seg(self, output_nc, n_downsampling, ngf, norm_layer, use_bias, growth_rate, block_config, num_init_features, bn_size, drop_rate, num_classes, num_times):    # seg TODO
        dense_ngf = self.get_dense_ngf(growth_rate, block_config, num_init_features, bn_size, drop_rate, num_classes)
        dense_downsampling = 2  # 因为缩小了5倍，改为上采样两次
        for i in range(dense_downsampling):
            if i==0:
                self.decoder_convTran_dict_seg['decoder_convTran%d' % (i + 1)] = nn.ConvTranspose2d(int(dense_ngf * num_times),
                                                                                                    int(dense_ngf),
                                                                                                    kernel_size=3,
                                                                                                    stride=2, padding=1,
                                                                                                    output_padding=1,
                                                                                                    bias=use_bias).type(
                    self.dtype)
                self.decoder_norm_dict_seg['decoder_norm%d' % (i + 1)] = norm_layer(int(dense_ngf)).type(self.dtype)
                self.decoder_relu_dict_seg['decoder_relu%d' % (i + 1)] = nn.ReLU(True).type(self.dtype)
            else:
                self.decoder_convTran_dict_seg['decoder_convTran%d' % (i + 1)] = nn.ConvTranspose2d(int(dense_ngf * 2),
                                                                                                    int(dense_ngf),
                                                                                                    kernel_size=3,
                                                                                                    stride=2, padding=1,
                                                                                                    output_padding=1,
                                                                                                    bias=use_bias).type(
                    self.dtype)
                self.decoder_norm_dict_seg['decoder_norm%d' % (i + 1)] = norm_layer(int(dense_ngf)).type(self.dtype)
                self.decoder_relu_dict_seg['decoder_relu%d' % (i + 1)] = nn.ReLU(True).type(self.dtype)

            if i!= (dense_downsampling-1):
                dense_ngf = dense_ngf // 2
        self.last_dense_ngf_seg = dense_ngf

    def get_encoder(self, input_nc, n_downsampling, ngf, norm_layer, use_dropout, n_blocks, padding_type, use_bias, growth_rate, block_config, num_init_features, bn_size, drop_rate, num_classes):
        # First convolution
        model =[]


        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate,
                                drop_rate=drop_rate)
            model += block
            num_features = num_features + num_layers * growth_rate

            # 使用没有avgpool的transition，保持经过denseblock后，输入的通道和大小都不变
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=num_features // 4)  # 256/4=64,64是初始通道，256是经过一个block后得到的升了的channel
                model += trans
                num_features = num_features // 4


        return nn.Sequential(*model)

    def get_decoder(self, output_nc, n_downsampling, ngf, norm_layer, use_bias, growth_rate, block_config, num_init_features, bn_size, drop_rate, num_classes):
        model = []

        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            if i!=(len(block_config)-1):    # i不是最后一个
                num_features=(num_features+num_layers*growth_rate)//2
            else:
                num_features = (num_features + num_layers * growth_rate)

        dense_ngf = num_features
        dense_downsampling=5    # 因为缩小了5倍

        for i in range(dense_downsampling):
            model += [nn.ConvTranspose2d(int(dense_ngf*2), int(dense_ngf), kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
                      norm_layer(int(dense_ngf)),
                      nn.ReLU(True)]
            dense_ngf=dense_ngf//2


        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(dense_ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        return nn.Sequential(*model)

    def forward(self, inp): # inp:torch.Size([1, 5, 200, 200])
        # split data
        img = inp[:, :self.input_nc, :, :]  # (B, CX, W, H) img:torch.Size([1, 3, 200, 200])
        segs = inp[:, self.input_nc:, :, :]  # (B, CA, W, H) segs:torch.Size([1, 2, 200, 200])
        mean = (segs + 1).mean(0).mean(-1).mean(-1)
        if mean.sum() == 0:
            mean[0] = 1  # forward at least one segmentation

        # run encoder
        # enc_img = self.encoder_img(img) # enc_img:torch.Size([1, 1024, 6, 6])

        conv0_out=self.conv0(img)   # conv0_out:torch.Size([1, 64, 100, 100])
        norm0_out=self.norm0(conv0_out) # norm0_out:
        relu0_out=self.relu0(norm0_out)
        maxpool0_out=self.maxpool0(relu0_out)
        enc_img = self.encoder_img(maxpool0_out) # enc_img:torch.Size([1, 1024, 6, 6])

        enc_segs = list()
        for i in range(segs.size(1)):   # i = 0,1循环两次
            if mean[i] > 0:  # skip empty segmentation
                seg = segs[:, i, :, :].unsqueeze(1)
                conv0_out = self.conv0_seg(seg)  # conv0_out:torch.Size([1, 64, 100, 100]) # TODO
                norm0_out = self.norm0(conv0_out)  # norm0_out:
                relu0_out = self.relu0(norm0_out)
                maxpool0_out = self.maxpool0(relu0_out)
                enc_segs.append(self.encoder_seg(maxpool0_out))  # 每次append的encoder后的seg的torch.Size([1, 1024, 6, 6])，总共append两次
        enc_segs = torch.cat(enc_segs)  # enc_segs:torch.Size([2, 1024, 6, 6])
        enc_segs_sum = torch.sum(enc_segs, dim=0, keepdim=True)  # aggregated set feature

        # run decoder
        feat = torch.cat([enc_img, enc_segs_sum], dim=1)    # feat：torch.Size([1, 2048, 6, 6]) # TODO


        decoder_convTran1_out = self.decoder_convTran_dict['decoder_convTran1'](feat)
        decoder_norm1_out = self.decoder_norm_dict['decoder_norm1'](decoder_convTran1_out)
        decoder_relu1_out = self.decoder_relu_dict['decoder_relu1'](decoder_norm1_out)

        decoder_convTran2_out = self.decoder_convTran_dict['decoder_convTran2'](decoder_relu1_out)
        decoder_norm2_out = self.decoder_norm_dict['decoder_norm2'](decoder_convTran2_out)
        decoder_relu2_out = self.decoder_relu_dict['decoder_relu2'](decoder_norm2_out)

        # decoder_convTran3_out = self.decoder_convTran_dict['decoder_convTran3'](decoder_relu2_out)
        # decoder_norm3_out = self.decoder_norm_dict['decoder_norm3'](decoder_convTran3_out)
        # decoder_relu3_out = self.decoder_relu_dict['decoder_relu3'](decoder_norm3_out)
        #
        # decoder_convTran4_out = self.decoder_convTran_dict['decoder_convTran4'](decoder_relu3_out)
        # decoder_norm4_out = self.decoder_norm_dict['decoder_norm4'](decoder_convTran4_out)
        # decoder_relu4_out = self.decoder_relu_dict['decoder_relu4'](decoder_norm4_out)
        #
        # decoder_convTran5_out = self.decoder_convTran_dict['decoder_convTran5'](decoder_relu4_out)
        # decoder_norm5_out = self.decoder_norm_dict['decoder_norm5'](decoder_convTran5_out)
        # decoder_relu5_out = self.decoder_relu_dict['decoder_relu5'](decoder_norm5_out)

        reflec_out = self.reflec(decoder_relu2_out)
        convlast_out = self.convlast(reflec_out)
        tanh_out = self.tanh(convlast_out)

        out = [tanh_out]
        # out = tanh_out
        # out = [self.decoder_img(feat)]


        idx = 0
        for i in range(segs.size(1)):
            if mean[i] > 0:
                enc_seg = enc_segs[idx].unsqueeze(0)  # (1, ngf, w, h)
                idx += 1  # move to next index
                feat = torch.cat([enc_seg, enc_img, enc_segs_sum], dim=1)   # feat:torch.Size([1, 3072, 7, 7])

                decoder_convTran1_out = self.decoder_convTran_dict_seg['decoder_convTran1'](feat)
                decoder_norm1_out = self.decoder_norm_dict_seg['decoder_norm1'](decoder_convTran1_out)
                decoder_relu1_out = self.decoder_relu_dict_seg['decoder_relu1'](decoder_norm1_out)

                decoder_convTran2_out = self.decoder_convTran_dict_seg['decoder_convTran2'](decoder_relu1_out)
                decoder_norm2_out = self.decoder_norm_dict_seg['decoder_norm2'](decoder_convTran2_out)
                decoder_relu2_out = self.decoder_relu_dict_seg['decoder_relu2'](decoder_norm2_out)

                # decoder_convTran3_out = self.decoder_convTran_dict_seg['decoder_convTran3'](decoder_relu2_out)
                # decoder_norm3_out = self.decoder_norm_dict_seg['decoder_norm3'](decoder_convTran3_out)
                # decoder_relu3_out = self.decoder_relu_dict_seg['decoder_relu3'](decoder_norm3_out)
                #
                # decoder_convTran4_out = self.decoder_convTran_dict_seg['decoder_convTran4'](decoder_relu3_out)
                # decoder_norm4_out = self.decoder_norm_dict_seg['decoder_norm4'](decoder_convTran4_out)
                # decoder_relu4_out = self.decoder_relu_dict_seg['decoder_relu4'](decoder_norm4_out)
                #
                # decoder_convTran5_out = self.decoder_convTran_dict_seg['decoder_convTran5'](decoder_relu4_out)
                # decoder_norm5_out = self.decoder_norm_dict_seg['decoder_norm5'](decoder_convTran5_out)
                # decoder_relu5_out = self.decoder_relu_dict_seg['decoder_relu5'](decoder_norm5_out)


                reflec_out = self.reflec(decoder_relu2_out)
                convlast_out = self.convlast_seg(reflec_out)    # 注意seg输出通道为1
                tanh_out = self.tanh(convlast_out)

                out += [tanh_out]

                # out += tanh_out
                # out += [self.decoder_seg(feat)]

            else:
                out += [segs[:, i, :, :].unsqueeze(1)]  # skip empty segmentation
        # tmp=torch.cat(out, dim=1)
        return torch.cat(out, dim=1)



