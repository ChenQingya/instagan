import sys
import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import torch


class UnalignedSegDataset(BaseDataset):
	def name(self):
		return 'UnalignedSegDataset'

	@staticmethod
	def modify_commandline_options(parser, is_train):
		return parser

	def initialize(self, opt):
		self.opt = opt
		self.root = opt.dataroot
		# `phase`就是对应例如`edges2shoes`文件夹下的子文件的名称（这里没有edges2shoes），这里选择`val`
		self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')# eg. self.dir_A is "datasets/shp2gir_coco/sampleA"
		self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')# eg. self.dir_A is "datasets/shp2gir_coco/sampleB"
		self.max_instances = 20  # default: 20
		self.seg_dir = 'seg'  # default: 'seg'

		# eg. obtain all the images' paths of
		# "datasets/shp2gir_coco/sampleA" or "datasets/shp2gir_coco/sampleB"
		self.A_paths = sorted(make_dataset(self.dir_A))
		self.B_paths = sorted(make_dataset(self.dir_B))
		self.A_size = len(self.A_paths)
		self.B_size = len(self.B_paths)
		self.transform = get_transform(opt)

	def fixed_transform(self, image, seed):
		random.seed(seed)
		return self.transform(image)

	def read_segs(self, seg_path, seed):
		segs = list()
		# max_instances means an image and its many segs
		# (the number of segs is not more than max_instances)
		# the image and its segs are as input of the model
		for i in range(self.max_instances):
			# eg. seg_path is "datasets/trainA_seg/0.png"
			# then path is "datasets/trainA_seg/0_0.png" or "datasets/trainA_seg/0_1.png" etc.
			path = seg_path.replace('.png', '_{}.png'.format(i))
			# confirm whether the file("datasets/trainA_seg/0_1.png") exits
			# if exist
			if os.path.isfile(path):
				# PIL image.convert. L means convert it to 灰度图
				# refer:https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.convert
				seg = Image.open(path).convert('L')
				# 转成固定大小的seg
				seg = self.fixed_transform(seg, seed)
				segs.append(seg)
			# if not exist
			else:
				# 若不存在，则生成的seg每个像素的值为-1
				segs.append(-torch.ones(segs[0].size()))
		# 将所有segs拼接 cat函数
		return torch.cat(segs)

	# Map-style datasets,A map-style dataset is one that implements
	# the __getitem__() and __len__() protocols,
	# and represents a map from (possibly non-integral) indices/keys to data samples.
	# refer:https://pytorch.org/docs/stable/data.html#map-style-datasets
	def __getitem__(self, index):
		index_A = index % self.A_size
		# serial_batches means no shuffle
		if self.opt.serial_batches:
			index_B = index % self.B_size
		else:
			index_B = random.randint(0, self.B_size - 1)

		# eg.datasets/trainA/0.png(already exit)
		A_path = self.A_paths[index_A]
		B_path = self.B_paths[index_B]
		# self.seg_dir is 'seg', A_path is eg.datasets/trainA/0.png(already exit)
		# then A_seg_path datasets/trainA_seg/0.png(not exist)
		A_seg_path = A_path.replace('A', 'A_{}'.format(self.seg_dir))
		B_seg_path = B_path.replace('B', 'B_{}'.format(self.seg_dir))

		# eg.
		# code:
		# 		str="http://www.runoob.com/python/att-string-split.html"
		# 		print("0:%s"%str.split("/")[-1])
		# result:
		# 		0:att-string-split.html
		# therefore, if A_path is eg.datasets/trainA/0.png(already exit) then A_idx is 0
		A_idx = A_path.split('/')[-1].split('.')[0]
		B_idx = B_path.split('/')[-1].split('.')[0]

		# print('(A, B) = (%d, %d)' % (index_A, index_B))
		seed = random.randint(-sys.maxsize, sys.maxsize)

		A = Image.open(A_path).convert('RGB')
		B = Image.open(B_path).convert('RGB')
		A = self.fixed_transform(A, seed)
		B = self.fixed_transform(B, seed)

		#get 一组segs by A_seg_path
		A_segs = self.read_segs(A_seg_path, seed)
		B_segs = self.read_segs(B_seg_path, seed)

		if self.opt.direction == 'BtoA':
			input_nc = self.opt.output_nc
			output_nc = self.opt.input_nc
		else:
			input_nc = self.opt.input_nc
			output_nc = self.opt.output_nc

		if input_nc == 1:  # RGB to gray
			tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
			A = tmp.unsqueeze(0)
		if output_nc == 1:  # RGB to gray
			tmp = B[0, ...] * 0.299 + B[1, ...] * 0.587 + B[2, ...] * 0.114
			B = tmp.unsqueeze(0)

		# 注意，一张图对应一组seg
		# eg. return an image of domain A in "datasets/shp2gir_coco/sampleA" and its idx and its a batch of segs and its path
		# eg. and return an image of domain A in "datasets/shp2gir_coco/sampleA" and its idx and its a batch of  segs and its path
		return {'A': A, 'B': B,
				'A_idx': A_idx, 'B_idx': B_idx,
				'A_segs': A_segs, 'B_segs': B_segs,
				'A_paths': A_path, 'B_paths': B_path}

	def __len__(self):
		return max(self.A_size, self.B_size)
