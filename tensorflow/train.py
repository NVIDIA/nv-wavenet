import os
import tensorflow as tf
import numpy as np
from mel2samp_onehot import Mel2SampOnehot
from wavenet import WaveNet
import json
from WaveNetEstimator import WaveNetEstimator

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


filelist = '/home/will/pylon-wavenet/pytorch/shelby.json'


def load_data(filepath):
    with open(filelist) as f:
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

def train(argv):
    args = parser.parse_args(argv[1:])

    
    classifier = WaveNetEstimator(
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
