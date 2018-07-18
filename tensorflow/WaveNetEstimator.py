"""Estimator for wavenet model"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.estimator.export import export_output

from wavenet import WaveNet


def custom_model_fn(features, labels, mode, params):
    """Model function for custom WaveNetEsimator"""
    model = WaveNet(**params)
    if mode == tf.estimator.ModeKeys.PREDICT:
        logits = model((features, labels), training=False)
        predictions = {
            'logits': logits
        }
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            export_outputs={'upsampled': export_output.PredictOutput(predictions)}
            )

    logits = model((features, labels), training=True)
    logits = tf.transpose(logits, [0, 2, 1])
    labels = tf.one_hot(tf.cast(labels, dtype=tf.int32), 256)

    loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)
    metrics = {'loss': loss}
    tf.summary.scalar('loss', loss)

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
        warm_start_from=None
    ):
        config = tf.estimator.RunConfig(
            save_checkpoints_steps=1000
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
