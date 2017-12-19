from __future__ import print_function
import numpy as np
import tensorflow as tf
import tifffile as tif
import cv2
import random
import os
import glob
import argparse

_PREFIX = {
  'train': 'tr',
  'valid': 'va',
  'test': 'te'
}

parser = argparse.ArgumentParser()
parser.add_argument('--sample_dir', type=str, default='./data')

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def glob_dir(wildcard):
  return len(glob.glob(wildcard))

def convert_to(name, num, sample_dir, save_dir='.'):

  # Create a TFRecords writer
  frec = os.path.join(save_dir, name + '.tfrecords')
  print('Write %s' % frec)
  writer = tf.python_io.TFRecordWriter(frec)

  _c = 0
  while _c < num:
    fim = os.path.join(sample_dir, '%s_n%04d_im' % (_PREFIX[name],  _c) + '.jpg')
    flb = os.path.join(sample_dir, '%s_n%04d_lb' % (_PREFIX[name],  _c) + '.jpg')
    im = cv2.imread(fim)
    lb = cv2.imread(flb, 0)
    im_raw = im.tostring()
    lb_raw = lb.tostring()
    example = tf.train.Example(features=tf.train.Features(
      feature={
        'label_raw': _bytes_feature(lb_raw),
        'image_raw': _bytes_feature(im_raw),
      }
    ))
    writer.write(example.SerializeToString())
    _c += 1
  writer.close()

def main():
  FLAGS = parser.parse_args()
  num_samples = {
    k: glob_dir(os.path.join(FLAGS.sample_dir, v + '*')) / 2 for k, v in _PREFIX.items()
  }
  print('Num of samples:')
  print(num_samples)
  for k, v in num_samples.items():
    if not v ==0:
      convert_to(k, v, FLAGS.sample_dir)


if __name__ == '__main__':
  main()
