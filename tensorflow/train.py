import os
import tensorflow as tf
import numpy as np
import json
import argparse

from mel2samp_onehot import Mel2SampOnehot
from wavenet import WaveNet
from WaveNetEstimator import WaveNetEstimator

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


def load_data(filepath):
    with open(filepath) as f:
        data = f.read()
        config = json.loads(data)
        data_config = config["data_config"]
        prepare = Mel2SampOnehot(**data_config)
        dataset = prepare.preprocess()
        features, labels = dataset
        features = tf.convert_to_tensor(np.asarray(features), dtype=tf.float32)
        labels = tf.convert_to_tensor(np.asarray(labels))
    return features, labels


def train_input_fn(features, labels, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)
    return dataset.make_one_shot_iterator().get_next()


def load_params(filepath):
    with open(filepath) as f:
        data = f.read()
        config = json.loads(data)
        wavenet_config = config["wavenet_config"]
    return wavenet_config


def train(args):
    print(load_params(args.config))
    classifier = WaveNetEstimator(
        model_dir=args.model_dir,
        params=load_params(args.config)
    )

    features, labels = load_data(args.config)

    classifier.train(
        input_fn=lambda: train_input_fn(features,
                                        labels,
                                        int(args.batch_size)),
        steps=int(args.train_steps)
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train WaveNet model')
    parser.add_argument('--model_dir',
                        help='directory of model checkpoint')
    parser.add_argument('--config',
                        help='text file of paths to training files')
    parser.add_argument('--batch_size',
                        help='batch size')
    parser.add_argument('--train_steps',
                        help='number of training steps')

    args = parser.parse_args()
    train(args)
