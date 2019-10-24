# 理解seg，seg转换

import sys
from PIL import Image
import os

# 以下是在ubuntu上的图像进行测试-------------------------------------------------------------

path = "/media/zlz422/8846427F46426E4E/cqy/instagan/datasets/shp2gir_coco/trainA_seg/0_0.png"
tmp1 = Image.open(path)
print(tmp1)
#< PIL.PngImagePlugin.PngImageFile image mode = L size = 256x256 at 0x7F1777227320 >
tmp2 = Image.open(path).convert('L')
print(tmp2)
#< PIL.Image.Image image mode = L size = 256x256 at 0x7F1777227D68 >



import importlib
baseoptlib = importlib.import_module("options.base_options")
baseopt = None
for name, cls in baseoptlib.__dict__.items():  # 此处datasetlib.__dict__.items(),返回：dict_items([('__name__', 'data.unaligned_seg_dataset'), ('__doc__', None),省略}，
    # 其中有一项为('UnalignedSegDataset', <class 'data.unaligned_seg_dataset.UnalignedSegDataset'>)
    if name.lower() == "BaseOptions".lower():  # 保证cls是类UnalignedSegDataset，所以cls的类型必须是BaseDataset的子类
        baseopt = cls
type(baseopt)
#<class 'type'>
instance = baseopt()
type(instance)
#<class 'options.base_options.BaseOptions'>


import torchvision
from torchvision import transforms
def get_transform():
    transform_list = []

    osize = [220, 220]
    fsize = [200, 200]
    transform_list.append(transforms.Resize(osize, Image.BICUBIC))
    transform_list.append(transforms.RandomCrop(fsize))

    # # Modify transform to specify width and height,指定输入图像的高度和宽度，或者指定seg的高度和宽度
    # if opt.resize_or_crop == 'resize_and_crop':
    #     # osize = [opt.loadSizeH, opt.loadSizeW]
    #     # fsize = [opt.fineSizeH, opt.fineSizeW]
    #     osize = [220, 220]
    #     fsize = [200, 200]
    #     transform_list.append(transforms.Resize(osize, Image.BICUBIC))
    #     transform_list.append(transforms.RandomCrop(fsize))
    # # Original CycleGAN code
    # # if opt.resize_or_crop == 'resize_and_crop':
    # #     osize = [opt.loadSize, opt.loadSize]
    # #     transform_list.append(transforms.Resize(osize, Image.BICUBIC))
    # #     transform_list.append(transforms.RandomCrop(opt.fineSize))
    # # elif opt.resize_or_crop == 'crop':
    # #     transform_list.append(transforms.RandomCrop(opt.fineSize))
    # # elif opt.resize_or_crop == 'scale_width':
    # #     transform_list.append(transforms.Lambda(
    # #         lambda img: __scale_width(img, opt.fineSize)))
    # # elif opt.resize_or_crop == 'scale_width_and_crop':
    # #     transform_list.append(transforms.Lambda(
    # #         lambda img: __scale_width(img, opt.loadSize)))
    # #     transform_list.append(transforms.RandomCrop(opt.fineSize))
    # # elif opt.resize_or_crop == 'none':
    # #     transform_list.append(transforms.Lambda(
    # #         lambda img: __adjust(img)))
    # else:
    #     raise ValueError('--resize_or_crop %s is not a valid option.' % opt.resize_or_crop)
    # if opt.isTrain and not opt.no_flip:
    #     transform_list.append(transforms.RandomHorizontalFlip())
    transform_list += [transforms.ToTensor(),
                       #                       transforms.Lambda(lambda x: x.repeat(3, 1, 1)), #失败
                       transforms.Normalize([0.5],
                                            [0.5])]
    return transforms.Compose(transform_list)
transformlist = get_transform()
tmp1trans = transformlist(tmp1)
tmp2trans = transformlist(tmp2)
tmplist = []
tmplist.append(tmp1trans)
tmplist.append(tmp2trans)
print(tmplist)
# [tensor([[[-1., -1., -1.,  ..., -1., -1., -1.],
#          [-1., -1., -1.,  ..., -1., -1., -1.],
#          [-1., -1., -1.,  ..., -1., -1., -1.],
#          ...,
#          [-1., -1., -1.,  ..., -1., -1., -1.],
#          [-1., -1., -1.,  ..., -1., -1., -1.],
#          [-1., -1., -1.,  ..., -1., -1., -1.]]]), tensor([[[-1., -1., -1.,  ..., -1., -1., -1.],
#          [-1., -1., -1.,  ..., -1., -1., -1.],
#          [-1., -1., -1.,  ..., -1., -1., -1.],
#          ...,
#          [-1., -1., -1.,  ..., -1., -1., -1.],
#          [-1., -1., -1.,  ..., -1., -1., -1.],
#          [-1., -1., -1.,  ..., -1., -1., -1.]]])]


import torch
torch.cat(tmplist)
# tensor([[[-1., -1., -1.,  ..., -1., -1., -1.],
#          [-1., -1., -1.,  ..., -1., -1., -1.],
#          [-1., -1., -1.,  ..., -1., -1., -1.],
#          ...,
#          [-1., -1., -1.,  ..., -1., -1., -1.],
#          [-1., -1., -1.,  ..., -1., -1., -1.],
#          [-1., -1., -1.,  ..., -1., -1., -1.]],
#         [[-1., -1., -1.,  ..., -1., -1., -1.],
#          [-1., -1., -1.,  ..., -1., -1., -1.],
#          [-1., -1., -1.,  ..., -1., -1., -1.],
#          ...,
#          [-1., -1., -1.,  ..., -1., -1., -1.],
#          [-1., -1., -1.,  ..., -1., -1., -1.],
#          [-1., -1., -1.,  ..., -1., -1., -1.]]])


# 以下操作在mac上的图像进行测试------------------------------------------------------------
# 为了理解select_masks_decreasing函数中的mean(-1)
import torch

path = "/Users/chenqy/PycharmProjects/instagan/datasets/shp2gir_coco/trainA_seg/139_0.png"
segs = []

from PIL import Image
transform_list = []
from torchvision import transforms

seg = Image.open(path).convert('L')
import torchvision


def get_transform():
    transform_list = []
    osize = [220, 220]
    fsize = [200, 200]
    transform_list.append(transforms.Resize(osize, Image.BICUBIC))
    transform_list.append(transforms.RandomCrop(fsize))
    # # Modify transform to specify width and height,指定输入图像的高度和宽度，或者指定seg的高度和宽度
    # if opt.resize_or_crop == 'resize_and_crop':
    #     # osize = [opt.loadSizeH, opt.loadSizeW]
    #     # fsize = [opt.fineSizeH, opt.fineSizeW]
    #     osize = [220, 220]
    #     fsize = [200, 200]
    #     transform_list.append(transforms.Resize(osize, Image.BICUBIC))
    #     transform_list.append(transforms.RandomCrop(fsize))
    # # Original CycleGAN code
    # # if opt.resize_or_crop == 'resize_and_crop':
    # #     osize = [opt.loadSize, opt.loadSize]
    # #     transform_list.append(transforms.Resize(osize, Image.BICUBIC))
    # #     transform_list.append(transforms.RandomCrop(opt.fineSize))
    # # elif opt.resize_or_crop == 'crop':
    # #     transform_list.append(transforms.RandomCrop(opt.fineSize))
    # # elif opt.resize_or_crop == 'scale_width':
    # #     transform_list.append(transforms.Lambda(
    # #         lambda img: __scale_width(img, opt.fineSize)))
    # # elif opt.resize_or_crop == 'scale_width_and_crop':
    # #     transform_list.append(transforms.Lambda(
    # #         lambda img: __scale_width(img, opt.loadSize)))
    # #     transform_list.append(transforms.RandomCrop(opt.fineSize))
    # # elif opt.resize_or_crop == 'none':
    # #     transform_list.append(transforms.Lambda(
    # #         lambda img: __adjust(img)))
    # else:
    #     raise ValueError('--resize_or_crop %s is not a valid option.' % opt.resize_or_crop)
    # if opt.isTrain and not opt.no_flip:
    #     transform_list.append(transforms.RandomHorizontalFlip())
    transform_list += [transforms.ToTensor(),
                       #                       transforms.Lambda(lambda x: x.repeat(3, 1, 1)), #失败
                       transforms.Normalize([0.5],
                                            [0.5])]
    return transforms.Compose(transform_list)


transformlist = get_transform()
segtransresult = transformlist(seg)
segtransresult
# Out[17]:
# tensor([[[-1., -1., -1., ..., -1., -1., -1.],
#          [-1., -1., -1., ..., -1., -1., -1.],
#          [-1., -1., -1., ..., -1., -1., -1.],
#          ...,
#          [-1., -1., -1., ..., -1., -1., -1.],
#          [-1., -1., -1., ..., -1., -1., -1.],
#          [-1., -1., -1., ..., -1., -1., -1.]]])
segs.append(segtransresult)
segtransresult.size()
# Out[19]: torch.Size([1, 200, 200])
seg.size
# Out[20]: (256, 256)
# segs加了21次segtransresult
for i in range(20):
    segs.append(segtransresult)

segscat = torch.cat(segs)
segscat.size()
# Out[25]: torch.Size([21, 200, 200])
originseg = Image.open(path)
originseg.size
# Out[27]: (256, 256)
meansegsfirst = segscat.mean(-1)            # 一次mean操作，从torch.Size([3, 200, 200])变成torch.Size([3, 200])
meansegsfirst.size()
# Out[29]: torch.Size([21, 200])
meansegsecond = segscat.mean(-1).mean(-1)   # 两次mean操作，从torch.Size([3, 200, 200])变成torch.Size([3])
meansegsecond.size()
# Out[31]: torch.Size([21])

# 以下是error的部分-------------------------------------------
# 为了理解select_mask()
segs_batch = segscat
ret = list()
insmax = 4

for singleseg in segs_batch:
    mean = singleseg.mean(-1).mean(-1)
    m, i = mean.topk(insmax)
    ret.append(singleseg[i, :, :])

# 出错在：
# m, i = mean.topk(insmax)
# RuntimeError: selected index k out of range

for singleseg in segs_batch:
    mean = singleseg.mean(-1).mean(-1)
    print(mean.size())

# torch.Size([])
# torch.Size([])
# torch.Size([])
# torch.Size([])
# torch.Size([])
# torch.Size([])
# torch.Size([])
# torch.Size([])
# torch.Size([])
# torch.Size([])
# torch.Size([])
# torch.Size([])
# torch.Size([])
# torch.Size([])
# torch.Size([])
# torch.Size([])
# torch.Size([])
# torch.Size([])
# torch.Size([])
# torch.Size([])
# torch.Size([])
for singleseg in segs_batch:
    print(singleseg.size())

# torch.Size([200, 200])
# torch.Size([200, 200])
# torch.Size([200, 200])
# torch.Size([200, 200])
# torch.Size([200, 200])
# torch.Size([200, 200])
# torch.Size([200, 200])
# torch.Size([200, 200])
# torch.Size([200, 200])
# torch.Size([200, 200])
# torch.Size([200, 200])
# torch.Size([200, 200])
# torch.Size([200, 200])
# torch.Size([200, 200])
# torch.Size([200, 200])
# torch.Size([200, 200])
# torch.Size([200, 200])
# torch.Size([200, 200])
# torch.Size([200, 200])
# torch.Size([200, 200])
# torch.Size([200, 200])