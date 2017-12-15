from __future__ import print_function
import numpy as np
import os
import tensorflow as tf

_DEPTH = 2
_CHANNELS = [8, 16, 32, 64, 128]

def unet_model(input, is_training=True):
  convs = []
  # Down sampling
  for c in _CHANNELS:
    for _ in range(_DEPTH):
      input = tf.layers.conv2d(inputs=input,
                              filters=c,
                              kernel_size=(3, 3),
                              padding='SAME',
                              activation=None)
      input = tf.layers.batch_normalization(inputs=input,
                                         training=is_training,
                                         center=True,
                                         scale=True,
                                         fused=True)
      input = tf.nn.relu(input)
    convs.append(input)
    if not c == _CHANNELS[-1]:
      input = tf.layers.max_pooling2d(inputs=input,
                                      pool_size=(2, 2),
                                      strides=(2, 2))

  convs = reversed(convs[:-1])
  # Up sampling
  for index, c in enumerate(reversed(_CHANNELS[:-1])):
    input = tf.image.resize_images(input, tf.shape(input)[1:3] * 2)
    input = tf.layers.conv2d(inputs=input,
                             filters=c,
                             kernel_size=(3, 3),
                             padding='SAME',
                             activation=None)
    input = tf.concat([input, convs.next()], axis=3)
    for _ in range(_DEPTH):
      input = tf.layers.conv2d(inputs=input,
                               filters=c,
                               kernel_size=3,
                               padding='SAME',
                               activation=None)
      input = tf.layers.batch_normalization(inputs=input,
                                            training=is_training,
                                            center=True,
                                            scale=True,
                                            fused=True)
      input = tf.nn.relu(input)

  logits = tf.layers.conv2d(inputs=input,
                            filters=2,
                            kernel_size=(3, 3),
                            padding='SAME',
                            activation=tf.nn.relu)

  return logits

if __name__ == '__main__':
  input = tf.constant(1.0, shape=[1, 512, 512, 3])
  _ = unet_model(input)
  

