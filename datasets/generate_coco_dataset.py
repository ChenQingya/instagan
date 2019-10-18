import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from pycocotools.coco import COCO

import torch
import torchvision.transforms as T


def main():
	parser = create_argument_parser()
	args = parser.parse_args()
	generate_coco_dataset(args)


def create_argument_parser():
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_root', type=str, default='datasets/COCO')
	parser.add_argument('--save_root', type=str, default='datasets/shp2gir_coco')
	parser.add_argument('--image_size', type=int, default=256, help='image size')
	parser.add_argument('--cat1', type=str, default='sheep', help='category 1')
	parser.add_argument('--cat2', type=str, default='giraffe', help='category 2')
	return parser

# eg. generate from datasets/COCO(the original datasets download from the net), and save in datasets/shp2gir_coco
# the difference is the generated has two domain images and their segs
def generate_coco_dataset(args):
	"""Generate COCO dataset (train/val, A/B)"""
	args.data_root = Path(args.data_root)
	args.save_root = Path(args.save_root)
	args.save_root.mkdir()

	generate_coco_dataset_sub(args, 'train', 'A', args.cat1)
	generate_coco_dataset_sub(args, 'train', 'B', args.cat2)
	generate_coco_dataset_sub(args, 'val', 'A', args.cat1)
	generate_coco_dataset_sub(args, 'val', 'B', args.cat2)


def generate_coco_dataset_sub(args, idx1, idx2, cat):
	"""
	Subroutine for generating COCO dataset
		- idx1: train/val
		- idx2: A/B
		- cat: category
	"""
	data_path = args.data_root / '{}2017'.format(idx1)
	anno_path = args.data_root / 'annotations/instances_{}2017.json'.format(idx1)	# eg. anno_path is "datasets/COCO/annotations/instances_train2017.json"
																					# or "datasets/COCO/annotations/instances_val2017.json"
	coco = COCO(anno_path)  # COCO API


	img_path = args.save_root / '{}{}'.format(idx1, idx2)		# eg. img_path is "datasets/shp2gir_coco/trainA" or "datasets/shp2gir_coco/trainB"
	seg_path = args.save_root / '{}{}_seg'.format(idx1, idx2)	# eg. img_path is "datasets/shp2gir_coco/trainA_seg" or "datasets/shp2gir_coco/trainB_seg"
	img_path.mkdir()											# they are empty, therefore mkdir()s
	seg_path.mkdir()

	cat_id = coco.getCatIds(catNms=cat)		# cat is "sheep" or "giraffe",get the category's id
	img_id = coco.getImgIds(catIds=cat_id)	# get the ids of sheep/giraffe images，获得所有绵羊的图片id，或者所有长颈鹿的图片id
	imgs = coco.loadImgs(img_id)			# 获得所有绵羊的图片(很多张)，或者所有长颈鹿的图片

	# tqdm表示进度条,progress
	# refer:https://tqdm.github.io/
	pb = tqdm(total=len(imgs))
	pb.set_description('{}{}'.format(idx1, idx2))
	for img in imgs:
		ann_ids = coco.getAnnIds(imgIds=img['id'], catIds=cat_id)	# get annotation'id
		anns = coco.loadAnns(ann_ids)								# get the annotation(many)

		count = 0
		for i in range(len(anns)):				# 真正从标签生成mask的地方。
			seg = coco.annToMask(anns[i])		# annotation to mask, the type is array now
			seg = Image.fromarray(seg * 255)	# turn the seg array to seg image,each pix multi 255. why?
			seg = resize(seg, args.image_size)	# resize the seg image
			# np.sum
			if np.sum(np.asarray(seg)) > 0:								# 保存seg
				seg.save(seg_path / '{}_{}.png'.format(pb.n, count))	# pb.n 表示？
				count += 1

		if count > 0:  # at least one instance exists
			img = Image.open(data_path / img['file_name'])
			img = resize(img, args.image_size)
			img.save(img_path / '{}.png'.format(pb.n))

		pb.update(1)
	pb.close()


def resize(img, size):
	return T.Compose([
		T.Resize(size),
		T.CenterCrop(size),
	])(img)


if __name__ == '__main__':
	main()
