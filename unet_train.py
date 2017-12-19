from __future__ import print_function
from unet_model import unet_model
import tensorflow as tf
import numpy as np
import sys
import argparse
import os

parser = argparse.ArgumentParser()

# Basic model parameters
parser.add_argument('--data_dir', type=str, default='.')

parser.add_argument('--model_dir', type=str, default='./unet_model')

parser.add_argument('--train_epochs', type=int, default=250)

parser.add_argument('--epochs_per_eval', type=int, default=10)

parser.add_argument('--batch_size', type=int, default=10)

parser.add_argument('--num_train', type=int, default=1000)

parser.add_argument('--num_valid', type=int, default=200)

_HEIGHT = 512
_WIDTH = 512
_CHANNELS = 3
_NUM_CLASSES = 2

_MEAN = None

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
        'label_raw': tf.FixedLenFeature([], tf.string),
      }
    )
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    image.set_shape([_HEIGHT * _WIDTH * _CHANNELS])
    label = tf.decode_raw(features['label_raw'], tf.uint8)
    label.set_shape([_HEIGHT * _WIDTH])
    image = tf.cast(image, tf.float32)
    label = tf.cast(label, tf.int32)
    return image, tf.one_hot(label, _NUM_CLASSES)

  dataset = tf.data.TFRecordDataset([filename])

  # Apply dataset transformations
  if is_training:
    dataset = dataset.shuffle(buffer_size=FLAGS.num_train)

  dataset = dataset.map(example_parser)
  dataset = dataset.map(
    lambda image, label: (preprocess_image(image, is_training), label)
  )
  dataset = dataset.prefetch(batch_size)
  dataset = dataset.repeat(num_epochs)
  dataset = dataset.batch(batch_size)
  iterator = dataset.make_one_shot_iterator()
  images, labels = iterator.get_next()

  return images, labels

def unet_model_fn(features, labels, mode):
  input = tf.reshape(features, [-1, _HEIGHT, _WIDTH, _CHANNELS])
  logits = unet_model(input, mode == tf.estimator.ModeKeys.TRAIN)

  predictions = {
    'classes': tf.argmax(input=logits, axis=3),
    'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
  }

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Process label reshaping after defining PREDICT mode estimator
  # to avoid error when predicting
  labels = tf.reshape(labels, [-1, _HEIGHT, _WIDTH, _NUM_CLASSES])

  # Show predicted mask image && ground truth mask image
  pd_label = tf.expand_dims(predictions['classes'], axis=-1) * 255
  tf.summary.image('pd_label',
                   tf.cast(pd_label, tf.uint8),
                   max_outputs=4)
  gt_label = tf.expand_dims(tf.argmax(input=labels, axis=3), axis=-1) * 255
  tf.summary.image('gt_label',
                   tf.cast(gt_label, tf.uint8),
                   max_outputs=4)

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
  train_file = os.path.join(FLAGS.data_dir, 'train.tfrecords')
  valid_file = os.path.join(FLAGS.data_dir, 'valid.tfrecords')

  # Manage GPU usages
  config_proto = tf.ConfigProto()
  config_proto.gpu_options.allow_growth = True
  config = tf.estimator.RunConfig(session_config=config_proto)

  # Create the Estimator
  unet_classifier = tf.estimator.Estimator(
    model_fn=unet_model_fn, model_dir=FLAGS.model_dir, config=config
  )

  # Set up training hook that log the training accuracy every 100 steps
  tensors_to_log = {
    'train_accuracy': 'train_accuracy'
  }
  logging_hook = tf.train.LoggingTensorHook(
    tensors=tensors_to_log, every_n_iter=100
  )

  # Train the model
  for i in range(FLAGS.train_epochs // FLAGS.epochs_per_eval):
    unet_classifier.train(
      input_fn=lambda: input_fn(
        True, train_file, FLAGS.batch_size, FLAGS.epochs_per_eval
      ),
      hooks=[logging_hook]
    )

    # Evaluate the model and print results
    eval_results = unet_classifier.evaluate(
      input_fn=lambda: input_fn(
        False, valid_file, FLAGS.batch_size
      )
    )
    print('Finished %d epochs, evaluation results:\n\t%s' %
          (i * FLAGS.epochs_per_eval, eval_results))


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(argv=[sys.argv[0]] + unparsed)