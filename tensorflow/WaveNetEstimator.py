"""Estimator for wavenet model"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import tensorflow as tf
import numpy as np
import json
import os

from wavenet import WaveNet
from mel2samp_onehot import Mel2SampOnehot

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=2, type=int, help='batch size')
parser.add_argument('--train_steps', default=1000000, type=int,
 
 
                    help='number of training steps')
class WaveNetEstimator(object):

    def __init__(self, model_dir, params):
        self.model_dir = model_dir
        self.params = params
        self.config = tf.estimator.RunConfig(
            save_checkpoints_secs=None,
            save_checkpoints_steps=1000
        )


    def model_fn(features, labels, mode, params):
        """Model function for custom WaveNetEsimator"""
        model = WaveNet(**params)
        logits = model((features, labels))
        logits = tf.transpose(logits, [0, 2, 1])

        labels = tf.one_hot(tf.cast(labels, dtype=tf.int32), 256)
        if mode == tf.estimator.ModeKeys.PREDICT:
            predictions = {
                'probabilities':  tf.nn.softmax(logits),
                'logits': logits
            }

        loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)

        accuracy = tf.metrics.accuracy(labels=labels,
                                    predictions=logits)
        metrics = {'accuracy': accuracy,
                'loss': loss}
        tf.summary.scalar('accuracy', accuracy)
        tf.summary.scalar('loss', loss)

        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(
                mode, loss=loss, eval_metric_ops=metrics
            )
        assert mode == tf.estimator.ModeKeys.TRAIN
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)



    def train_input_fn(features, labels, batch_size):
        dataset = tf.data.Dataset.from_tensor_slices((features, labels))
        dataset = dataset.shuffle(1000).repeat().batch(batch_size)
        return dataset.make_one_shot_iterator().get_next()


    def export_weights(self):
        model = {}


    def build_model(self):
        
        classifier = tf.estimator.Estimator(
            model_fn = self.model_fn,
            model_dir = self.model_dir,
            params = self.params,
            config = self.config
        )
        return classifier



def main(argv):
    args = parser.parse_args(argv[1:])

    filelist = '/home/will/pylon-wavenet/pytorch/shelby.json'
    print("Loading data....")
    with open(filelist) as f:
        data = f.read()
        config = json.loads(data)
        data_config = config["data_config"]
        prepare = Mel2SampOnehot(**data_config)
        dataset = prepare.preprocess()
        features, labels = dataset
        features = tf.convert_to_tensor(np.asarray(features), dtype=tf.float32)
        labels = tf.convert_to_tensor(np.asarray(labels))
    print("Load complete....")

    config = tf.estimator.RunConfig(
        
    )

    classifier = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir='./logs',
        params={
            'n_in_channels': 256,
            'n_layers': 16,
            'max_dilation': 128,
            'n_residual_channels': 64,
            'n_skip_channels': 256,
            'n_out_channels': 256,
            'n_cond_channels': 80,
            'upsamp_window': 1024,
            'upsamp_stride': 256
        }
    )

    classifier.train(
        input_fn=lambda: train_input_fn(features, labels, args.batch_size),
        steps=args.train_steps
    )


if __name__ == '__main__':

    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
