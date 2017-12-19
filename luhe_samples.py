'''Generate num_train/num_valid/num_test samples
with respect to rs file and lb file

Give a none-zero number to specific num_mode and a pair of
corresponding files

Every train sample will be named as tr_n0001_im.jpg/tr_n0001_lb.jpg
validation and test samples in similar format

Generated results will be save into defined save_dir
'''
from __future__ import print_function
import numpy as np
import tensorflow as tf
import tifffile as tif
import cv2
import random
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--num_train', type=int, default=0)

parser.add_argument('--num_valid', type=int, default=0)

parser.add_argument('--num_test', type=int, default=0)

parser.add_argument('--rs_file', type=str,
                    default='../sample_library/luhe.tif')

parser.add_argument('--lb_file', type=str,
                    default='../sample_library/luhe_buildings_classes.tif')

parser.add_argument('--save_dir', type=str, default='./data')


_SIZE = 512
_CHANNELS = 3

def cut_patches(image, label, mode, num, augmentation=False, directory='.'):
  # TODO:
  # 1. normalize && zero center
  # 2. rotate cutting

  name_mapping = {
    'train': 'tr',
    'valid': 'va',
    'test': 'te'
  }
  _h = image.shape[0]
  _w = image.shape[1]

  _c = 0
  _thr = 0.001
  while _c < num:
    xs = random.randint(0, _h - _SIZE)
    ys = random.randint(0, _w - _SIZE)
    im = image[xs:xs+_SIZE, ys:ys+_SIZE, :_CHANNELS]
    lb = label[xs:xs+_SIZE, ys:ys+_SIZE]
    if float(np.sum(lb, axis=(0, 1))) / _SIZE**2 < _thr:
      continue
    fim = os.path.join(directory, '%s_n%04d_im' % (name_mapping[mode],  _c) + '.jpg')
    flb = os.path.join(directory, '%s_n%04d_lb' % (name_mapping[mode],  _c) + '.jpg')
    cv2.imwrite(fim, im)
    cv2.imwrite(flb, lb * 255)
    _c += 1

def main():
  FLAGS = parser.parse_args()
  image = tif.imread(FLAGS.rs_file)
  label = tif.imread(FLAGS.lb_file)
  print('Loaded rs file and lb file')

  num_images = {
    'train': FLAGS.num_train,
    'valid': FLAGS.num_valid,
    'test': FLAGS.num_test
  }
  if not os.path.exists(FLAGS.save_dir):
    os.makedirs(FLAGS.save_dir)
  for k, v in num_images.items():
    if not v == 0:
      cut_patches(image, label, mode=k, num=v, augmentation=True, directory=FLAGS.save_dir)

if __name__ == '__main__':
  main()
