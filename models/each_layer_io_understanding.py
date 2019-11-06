# 为了理解instagan中经过resnetgenerator或者resnetsetgenerator
# 每一层的输入输出
# /home/zlz422/anaconda3/envs/cqy-torch0.4.0/bin/python /home/zlz422/pycharm-community-2018.1.4/helpers/pydev/pydevconsole.py 43631 34771
import sys; print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(['/media/zlz422/8846427F46426E4E/cqy/instagan'])
# PyDev console: starting.
# Python 3.6.8 |Anaconda, Inc.| (default, Dec 30 2018, 01:22:34)
# [GCC 7.3.0] on linux
import torch
inputshape=torch.Size([1,3,200,200])            # 原始图片的大小

length=len(inputshape)                          # 4维
length
# 4
# resnetsetgenerator:encoder----------------------------------

import torch.nn as nn
m=nn.ReflectionPad2d(3)
input=torch.randn(1,3,200,200)                  # 原始图片的大小
result=m(input)
print(result.size())
torch.Size([1, 3, 206, 206])

n=nn.Conv2d(3, 64, kernel_size=7, padding=0)
nresult=n(result)
print(nresult.size())
torch.Size([1, 64, 200, 200])

x=nn.Conv2d(64 *1, 64 * 1 * 2, kernel_size=3,stride=2, padding=1)
xresult=x(nresult)
print(xresult.size())
torch.Size([1, 128, 100, 100])

y=nn.Conv2d(64 * 2, 64 * 2 * 2, kernel_size=3,stride=2, padding=1)
yresult=y(xresult)
print(yresult.size())
torch.Size([1, 256, 50, 50])

mult = 2**2
a=nn.ReflectionPad2d(1)
aresult=a(yresult)
print(aresult.size())
torch.Size([1, 256, 52, 52])

b=nn.ReplicationPad2d(1)
bresult=b(yresult)
print(bresult.size())
torch.Size([1, 256, 52, 52])

ac=nn.Conv2d(64*4, 64*4, kernel_size=3, padding=0)
acresult=ac(aresult)
print(acresult.size())
torch.Size([1, 256, 50, 50])

c=nn.Conv2d(64*4, 64*4, kernel_size=3, padding=1)
cresult=c(yresult)
print(cresult.size())
torch.Size([1, 256, 50, 50])

# resnetsetgenerator:decoder------------------------------------------------

decoderinput=torch.randn(1,512,50,50)       # decoder的img输入

da=nn.ConvTranspose2d(128 * 4, int(128 * 4 / 2), kernel_size=3, stride=2, padding=1, output_padding=1)
daresult=da(decoderinput)
print(daresult.size())
torch.Size([1, 256, 100, 100])

db=nn.ConvTranspose2d(128 * 2, int(128 * 2 / 2), kernel_size=3, stride=2, padding=1, output_padding=1)
dbresult=db(daresult)
print(dbresult.size())
torch.Size([1, 128, 200, 200])

r=nn.ReflectionPad2d(3)
rresult=r(dbresult)
print(rresult.size())
torch.Size([1, 128, 206, 206])

conv=nn.Conv2d(128, 3, kernel_size=7, padding=0)
convresult=conv(rresult)
print(convresult.size())
torch.Size([1, 3, 200, 200])

tanh=nn.Tanh()
tanhresult=tanh(convresult)
print(tanhresult.size())
torch.Size([1, 3, 200, 200])

# densenet:encoder----------------------------------------------
from torchvision.models import densenet

tmp=torch.randn(1,3,200,200)
densemodel=densenet.DenseNet()
outputs=densemodel(tmp)


from collections import OrderedDict
num_init_features=64
first=nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2,
                                padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))
i=0
num_layers=6
denselayer=nn.Sequential(OrderedDict([]))

num_input_features=64
growth_rate=32
bn_size=4
denselayer=nn.Sequential(OrderedDict([('norm1', nn.BatchNorm2d(num_input_features)),('relu1', nn.ReLU(inplace=True)),('conv1', nn.Conv2d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1,
                                           bias=False)),('norm2', nn.BatchNorm2d(bn_size * growth_rate)),('relu2', nn.ReLU(inplace=True)),('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1,
                                           bias=False))]))

inputorigin=torch.randn(1,3,200,200)
firstout=first(inputorigin)
print(firstout.size())
torch.Size([1, 64, 50, 50])

densenlayerout=denselayer(firstout)
print(densenlayerout.size())
torch.Size([1, 32, 50, 50])
catout=torch.cat([firstout,densenlayerout],1)
print(catout.size())
torch.Size([1, 96, 50, 50])


num_features=num_init_features
num_features = num_features + num_layers * growth_rate

num_output_features=num_features // 2
translayer=nn.Sequential(OrderedDict([('norm', nn.BatchNorm2d(num_input_features)),('relu', nn.ReLU(inplace=True)),('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False)),('pool', nn.AvgPool2d(kernel_size=2, stride=2))]))


num_input_features=num_features
num_output_features=num_features // 2
translayer=nn.Sequential(OrderedDict([('norm', nn.BatchNorm2d(num_input_features)),('relu', nn.ReLU(inplace=True)),('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False)),('pool', nn.AvgPool2d(kernel_size=2, stride=2))]))




denselayer0=nn.Sequential(OrderedDict([('norm1', nn.BatchNorm2d(64)),('relu1', nn.ReLU(inplace=True)),('conv1', nn.Conv2d(64, 4 *
                                           32, kernel_size=1, stride=1,
                                           bias=False)),('norm2', nn.BatchNorm2d(4 * 32)),('relu2', nn.ReLU(inplace=True)),('conv2', nn.Conv2d(4 * 32, 32,
                                           kernel_size=3, stride=1, padding=1,
                                           bias=False))]))
denselayer0out=denselayer0(firstout)
print(denselayer0out.size())
torch.Size([1, 32, 50, 50])



denselayer1=nn.Sequential(OrderedDict([('norm1', nn.BatchNorm2d(64+1*32)),('relu1', nn.ReLU(inplace=True)),('conv1', nn.Conv2d(64+1*32, 4 *
                                           32, kernel_size=1, stride=1,
                                           bias=False)),('norm2', nn.BatchNorm2d(4 * 32)),('relu2', nn.ReLU(inplace=True)),('conv2', nn.Conv2d(4 * 32, 32,
                                           kernel_size=3, stride=1, padding=1,
                                           bias=False))]))
forward0=torch.cat([denselayer0out,firstout],dim=1)
denselayer1out=denselayer1(forward0)
print(denselayer1out.size())
torch.Size([1, 32, 50, 50])


trans0=nn.Sequential(OrderedDict([('norm', nn.BatchNorm2d(256)),('relu', nn.ReLU(inplace=True)),('conv', nn.Conv2d(256, 128,
                                          kernel_size=1, stride=1, bias=False)),('pool', nn.AvgPool2d(kernel_size=2, stride=2))]))

transinput=torch.randn(1,256,50,50)
trans0out=trans0(transinput)
print(trans0out.size())
torch.Size([1, 128, 25, 25])
avgpool=nn.AvgPool2d(2,2)
avgout=avgpool(torch.randn(1,128,50,50))
print(avgout.size())
torch.Size([1, 128, 25, 25])


convtmp=nn.Conv2d(256,128,kernel_size=1,stride=1,bias=False)
convout=convtmp(torch.randn(1,256,50,50))
print(convout.size())
torch.Size([1, 128, 50, 50])
denselayer0block2=nn.Sequential(OrderedDict([('norm1', nn.BatchNorm2d(128)),('relu1', nn.ReLU(inplace=True)),('conv1', nn.Conv2d(128, 4 *
                                           32, kernel_size=1, stride=1,
                                           bias=False)),('norm2', nn.BatchNorm2d(4 * 32)),('relu2', nn.ReLU(inplace=True)),('conv2', nn.Conv2d(4 * 32, 32,
                                           kernel_size=3, stride=1, padding=1,
                                           bias=False))]))
denselayer0block2out=denselayer0block2(trans0out)
print(denselayer0block2out.size())
torch.Size([1, 32, 25, 25])

# densenetgenerator:decoder-------------------------------------------------------
block_config=(6, 12, 24, 16)
len(block_config)-1
3

convtrantmp=nn.ConvTranspose2d(512+16*32, int((512+16*32)/ 2), kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)

convtrantmp=nn.ConvTranspose2d((512+16*32)*2, int((512+16*32)*2 / 2), kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
convtrantmpout=convtrantmp(torch.randn(1,2048,6,6))
print(convtrantmpout.size())
torch.Size([1, 1024, 12, 12])

# # 失败
# convtrantmp1=nn.ConvTranspose2d((512+16*32), int((512+16*32) / 2), kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
# convtrantmp1out=convtrantmp1(torch.randn(1,1024,12,12))
# print(convtrantmp1out.size())
# torch.Size([1, 512, 24, 24])
#
# convtrantmp1=nn.ConvTranspose2d((512+16*32), int((512+16*32) / 2), kernel_size=3, stride=2, padding=2, output_padding=1, bias=False)
# convtrantmp1out=convtrantmp1(torch.randn(1,1024,12,12))
# print(convtrantmp1out.size())
# torch.Size([1, 512, 22, 22])
# convtrantmp1=nn.ConvTranspose2d((512+16*32), int((512+16*32) / 2), kernel_size=3, stride=2, padding=1, output_padding=2, bias=False)
# convtrantmp1=nn.ConvTranspose2d((512+16*32), int((512+16*32) / 2), kernel_size=3, stride=2, padding=2, output_padding=2, bias=False, dilation=2)

# 成功
convtrantmp1=nn.ConvTranspose2d((512+16*32), int((512+16*32) / 2), kernel_size=4, stride=2, padding=1, output_padding=1, bias=False)
convtrantmp1out=convtrantmp1(torch.randn(1,1024,12,12))
print(convtrantmp1out.size())
torch.Size([1, 512, 25, 25])

convtrantmp2=nn.ConvTranspose2d(int((512+16*32)/2), int((512+16*32)/2/2), kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
convtrantmp2out=convtrantmp2(torch.randn(1,512,25,25))
print(convtrantmp2out.size())
torch.Size([1, 256, 50, 50])

convtrantmp3=nn.ConvTranspose2d(int((512+16*32)/2/2), int((512+16*32)/2/2/2), kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
convtrantmp3out=convtrantmp3(torch.randn(1,256,50,50))
print(convtrantmp3out.size())
torch.Size([1, 128, 100, 100])

convtrantmp4=nn.ConvTranspose2d(int((512+16*32)/2/2/2), int((512+16*32)/2/2/2/2), kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
convtrantmp4out=convtrantmp4(torch.randn(1,128,100,100))
print(convtrantmp4out.size())
torch.Size([1, 64, 200, 200])