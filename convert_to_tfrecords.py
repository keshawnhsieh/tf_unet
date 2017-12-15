from __future__ import print_function
import numpy as np
import tensorflow as tf
import tifffile as tif
import cv2
import random
import os

RS_FILE = '../sample_library/luhe.tif'
LB_FILE = '../sample_library/luhe_roadline_classes.tif'

# For visual purpose
PIC_SAVE_DIR = './data'

_NUM_IMAGES = {
  'train': 1000,
  'valid': 200,
}
_CHANNELS = 3
_SIZE = 512

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def convert_to(image, label, name, directory='.'):
  # TODO:
  # 1. normalize && zero center
  # 2. rotate cutting

  _h = image.shape[0]
  _w = image.shape[1]

  # Create a TFRecords writer
  frec = os.path.join(directory, name + '.tfrecords')
  print('Write %s' % frec)
  writer = tf.python_io.TFRecordWriter(frec)

  _c = 0
  _thr = 0.001
  while _c < _NUM_IMAGES[name]:
    xs = random.randint(0, _h - _SIZE)
    ys = random.randint(0, _w - _SIZE)
    im = image[xs:xs+_SIZE, ys:ys+_SIZE, :_CHANNELS]
    lb = label[xs:xs+_SIZE, ys:ys+_SIZE]
    if float(np.sum(lb, axis=(0, 1))) / _SIZE**2 < _thr:
      continue

    # Save 3 channels picture to disk, only for debug
    # fim = os.path.join(PIC_SAVE_DIR, 's%d_n%04d_im' % (_SIZE, _c) + '.jpg')
    # flb = os.path.join(PIC_SAVE_DIR, 's%d_n%04d_lb' % (_SIZE, _c) + '.jpg')
    # cv2.imwrite(fim, im)
    # cv2.imwrite(flb, lb * 255)

    im_raw = im.tostring()
    lb_raw = lb.tostring()
    example = tf.train.Example(features=tf.train.Features(
      feature={
        'height': _int64_feature(_h),
        'width': _int64_feature(_w),
        'depth': _int64_feature(_CHANNELS),
        'label_raw': _bytes_feature(lb_raw),
        'image_raw': _bytes_feature(im_raw),
      }
    ))
    writer.write(example.SerializeToString())
    _c += 1
  writer.close()

if __name__ == '__main__':
  image = tif.imread(RS_FILE)
  label = tif.imread(LB_FILE)
  print('Loaded rs file and lb file')
  convert_to(image, label, 'train')
  convert_to(image, label, 'valid')
