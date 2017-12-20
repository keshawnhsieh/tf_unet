from __future__ import print_function
from unet_model import unet_model
import tensorflow as tf
import numpy as np
import sys
import argparse
import os
import cv2

parser = argparse.ArgumentParser()

# Basic model parameters
parser.add_argument('--data_dir', type=str, default='.')

parser.add_argument('--model_dir', type=str, default='./unet_model')

parser.add_argument('--save_dir', type=str, default='./result')

parser.add_argument('--batch_size', type=int, default=10)

parser.add_argument('--gpu', type=str, default=None)

def set_gpus(kernel):
  os.environ['CUDA_VISIBLE_DEVICES'] = kernel

_HEIGHT = 512
_WIDTH = 512
_CHANNELS = 3
_NUM_CLASSES = 2

def preprocess_image(image, is_training):
  image = tf.reshape(image, shape=[_HEIGHT, _WIDTH, _CHANNELS])
  image = tf.image.per_image_standardization(image)
  image = tf.reshape(image, shape=[-1])
  return image

def input_fn(is_training, filename, batch_size=4, num_epochs=1):
  def example_parser(serialized_example):
    features = tf.parse_single_example(
      serialized_example,
      features={
        'image_raw': tf.FixedLenFeature([], tf.string),
        # 'label_raw': tf.FixedLenFeature([], tf.string),
      }
    )
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    image.set_shape([_HEIGHT * _WIDTH * _CHANNELS])
    # label = tf.decode_raw(features['label_raw'], tf.uint8)
    # label.set_shape([_HEIGHT * _WIDTH])
    image = tf.cast(image, tf.float32)
    # label = tf.cast(label, tf.int32)
    # return image, tf.one_hot(label, _NUM_CLASSES)
    return image

  dataset = tf.data.TFRecordDataset([filename])

  # Apply dataset transformations
  # if is_training:
  #   dataset = dataset.shuffle(buffer_size=_NUM_IMAGES['train'])

  dataset = dataset.map(example_parser)
  dataset = dataset.map(
    lambda image: preprocess_image(image, is_training)
  )
  dataset = dataset.prefetch(batch_size)
  dataset = dataset.repeat(num_epochs)
  dataset = dataset.batch(batch_size)
  iterator = dataset.make_one_shot_iterator()
  images = iterator.get_next()

  return images


def unet_model_fn(features, labels, mode):
  input = tf.reshape(features, [-1, _HEIGHT, _WIDTH, _CHANNELS])
  logits = unet_model(input, mode == tf.estimator.ModeKeys.TRAIN)

  predictions = {
    'classes': tf.argmax(input=logits, axis=3),
    'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
  }

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Process label reshape after defining PREDICT mode estimator
  # to avoid error when predicting
  labels = tf.reshape(labels, [-1, _HEIGHT, _WIDTH, _NUM_CLASSES])
  loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)

  # Configure the training op
  # TODO: use learning rate schedule
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
    train_op = optimizer.minimize(loss, tf.train.get_or_create_global_step())
  else:
    train_op = None

  accuracy = tf.metrics.accuracy(
    tf.argmax(labels, axis=3), predictions['classes']
  )
  metrics = {'accuracy': accuracy}

  # Create a tensor for logging purpose
  tf.identity(accuracy[1], name='train_accuracy')
  tf.summary.scalar('train_accuracy', accuracy[1])

  return tf.estimator.EstimatorSpec(
    mode=mode,
    predictions=predictions,
    loss=loss,
    train_op=train_op,
    eval_metric_ops=metrics
  )

def main(unused_argv):
  os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
  if FLAGS.gpu is not None:
    set_gpus(FLAGS.gpu)
  test_file = os.path.join(FLAGS.data_dir, 'test.tfrecords')

  # Manage GPU usages
  config_proto = tf.ConfigProto()
  config_proto.gpu_options.allow_growth = True
  config = tf.estimator.RunConfig(session_config=config_proto)

  # Create the Estimator
  unet_classifier = tf.estimator.Estimator(
    model_fn=unet_model_fn, model_dir=FLAGS.model_dir, config=config
  )

  # Excute predictions
  predictions = unet_classifier.predict(
    input_fn=lambda: input_fn(
      False, test_file, FLAGS.batch_size
    )
  )
  if not os.path.exists(FLAGS.save_dir):
    os.makedirs(FLAGS.save_dir)
  for i, p in enumerate(predictions):
    im = p['classes']
    name = os.path.join(FLAGS.save_dir, 'pd_n%04d_lb.jpg' % (i))
    cv2.imwrite(name, im)
    print('Saved prediction %s' % name)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(argv=[sys.argv[0]] + unparsed)