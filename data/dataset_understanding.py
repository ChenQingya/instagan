import torch
target_real_label = 1.0
torch.tensor(target_real_label)
# Out[6]: tensor(1.)


import importlib
dataset_filename = "data.unaligned_seg_dataset"
datasetlib = importlib.import_module(dataset_filename)
for name, cls in datasetlib.__dict__.items():
    print(type(name), type(cls))

datasetlib.__dict__
# Out[12]: {'__name__': 'data.unaligned_seg_dataset',
#  '__doc__': None,
#  '__package__': 'data',
#  '__loader__': <_frozen_importlib_external.SourceFileLoader at 0x11a4f7c10>,省略,'UnalignedSegDataset': data.unaligned_seg_dataset.UnalignedSegDataset}

datasetlib.__dict__.items()
# Out[13]:
# dict_items([('__name__', 'data.unaligned_seg_dataset'), 省略])

import functools
functools.partial
# Out[16]: functools.partial
type(functools.partial)
# Out[17]: type

import torch.nn as nn
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
norm = 'instance'
norm_layer = get_norm_layer(norm_type=norm)
type(norm_layer)
# Out[26]: functools.partial
type(functools.partial)
# Out[27]: type

import torch
import torch.nn as nn
m = nn.ReflectionPad2d(2)
input = torch.arange(9, dtype=torch.float).reshape(1, 1, 3, 3)
input
# tensor([[[[0., 1., 2.],
#           [3., 4., 5.],
#           [6., 7., 8.]]]])
m(input)
# tensor([[[[8., 7., 6., 7., 8., 7., 6.],
#           [5., 4., 3., 4., 5., 4., 3.],
#           [2., 1., 0., 1., 2., 1., 0.],
#           [5., 4., 3., 4., 5., 4., 3.],
#           [8., 7., 6., 7., 8., 7., 6.],
#           [5., 4., 3., 4., 5., 4., 3.],
#           [2., 1., 0., 1., 2., 1., 0.]]]])
# using different paddings for different sides
m = nn.ReflectionPad2d((1, 1, 2, 0))
m(input)
# tensor([[[[7., 6., 7., 8., 7.],
#           [4., 3., 4., 5., 4.],
#           [1., 0., 1., 2., 1.],
#           [4., 3., 4., 5., 4.],
#           [7., 6., 7., 8., 7.]]]])