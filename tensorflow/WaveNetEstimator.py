"""Estimator for wavenet model"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from wavenet import WaveNet


<<<<<<< 0698a676dc79a2dcf96370cc8ad3dbb56931d96c
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
=======
def custom_model_fn(features, labels, mode, params):
    """Model function for custom WaveNetEsimator"""
    model = WaveNet(**params)
    logits = model((features, labels))
    logits = tf.transpose(logits, [0, 2, 1])
>>>>>>> Cleaned up class

    labels = tf.one_hot(tf.cast(labels, dtype=tf.int32), 256)
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'probabilities':  tf.nn.softmax(logits),
            'logits': logits
        }

    loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)

    metrics = {'loss': loss}

    tf.summary.scalar('loss', loss)

<<<<<<< 0698a676dc79a2dcf96370cc8ad3dbb56931d96c
    def build_model(self):
        
        classifier = tf.estimator.Estimator(
            model_fn = self.model_fn,
            model_dir = self.model_dir,
            params = self.params,
            config = self.config
=======
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=metrics
        )
    assert mode == tf.estimator.ModeKeys.TRAIN
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


class WaveNetEstimator(tf.estimator.Estimator):
    def __init__(
        self,
        model_dir=None,
        config=None,
        params=None,
        warm_start_from=None,
        dropout=None
    ):
        config = tf.estimator.RunConfig(
            save_checkpoints_steps=1000
>>>>>>> Cleaned up class
        )

        def _model_fn(features, labels, mode, config, params):
            return custom_model_fn(
                features=features,
                labels=labels,
                mode=mode,
                params=params,
            )

        super(WaveNetEstimator, self).__init__(
            model_fn=_model_fn, model_dir=model_dir, config=config,
            warm_start_from=warm_start_from, params=params
        )
