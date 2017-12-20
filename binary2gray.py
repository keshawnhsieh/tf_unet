'''Used to convert binary images with keywords in a specified directory to grayscale images

Given the specific directory and the prefix key word

Return the gray mask image with the same name in the same directory

Use this tool temporarily, I will update it to a more elegant way ASAP
'''

from __future__ import print_function
import glob
import cv2
import os
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--target_dir', type=str, default=None)

parser.add_argument('--prefix', type=str, default='te')

FLAGS = parser.parse_args()

if FLAGS.target_dir is None:
  print('Must specify the target directory')
  exit(0)

fl = glob.glob(os.path.join(FLAGS.target_dir, FLAGS.prefix + '*'))
for f in fl:
  if 'lb' in f:
    mask = cv2.imread(f, 0)
    cv2.imwrite(f, mask * 255)
print('Converted %d files' % len(fl))
