import os
import sys

import torch
import tensorflow as tf
import numpy as np
from tensorflow.python.saved_model import tag_constants
from tqdm import tqdm

import nv_wavenet
from wavenet_utils import MAX_WAV_VALUE, mu_law_decode_numpy


def mel_to_torch(mels):
    mels = mels.T
    mels = torch.tensor(mels).cuda(1)
    mels = torch.unsqueeze(mels, 0)
    return mels


def load_graph(session, path):
    tf.saved_model.loader.load(
        session,
        [tag_constants.SERVING],
        path
    )
    return session.graph


def load_utterances(path):
    with open(path, encoding='utf-8') as file:
        files = file.readlines()
        lines = [f.strip().split('|') for f in files]
        utterances = [line[6] for line in lines]
    return utterances

if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

    outdir = '/var/pylon/data/speech/pylon/tts/shelby'

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.4

    SUSHIBOT_CHARSET = ('_~ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
                        + '!\'(),-.:;?  ,abdefghijklmnoprstuvwzæðŋɑɔəɛɝɪʃʊʌʒˈˌːθ')

    sushi_path = '/var/pylon/models/shelby-sushibot/'
    wavenet_path = 'checkpoints/shelby_retrain/wavenet_135000'
    tts_file = '/var/pylon/data/speech/pylon/tts/shelby/tts-train.txt'

    utterances = load_utterances(tts_file)

    tf.reset_default_graph()
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    graph = load_graph(sess, sushi_path)
    sushibot_inputs = graph.get_tensor_by_name("data/inputs:0")
    sushibot_lengths = graph.get_tensor_by_name("data/input_lengths:0")
    prediction = graph.get_tensor_by_name("sushibot/prediction:0")

    model = torch.load(wavenet_path)['model'].cuda(1)
    wavenet = nv_wavenet.NVWaveNet(**model.export_weights())

    for i, utterance in enumerate(tqdm(utterances)):
        input_vector = [[SUSHIBOT_CHARSET.index(c)
                        if c in SUSHIBOT_CHARSET else 0
                        for c in utterance] + [SUSHIBOT_CHARSET.index('~')]]

        feed_dict = {sushibot_inputs: input_vector,
                     sushibot_lengths: [len(input_vector[0])]}

        mels = sess.run(prediction, feed_dict=feed_dict)
        mels = mels.reshape(-1, 80)
        np.save(
            os.path.join(outdir, 'mels/sushi-mel-{:05d}.npy'.format(i)), mels)

        mels = mel_to_torch(mels)

        cond_input = model.get_cond_input(mels)
        waveform = wavenet.infer(cond_input, 2)
        audio = mu_law_decode_numpy(waveform[0, :].cpu().numpy(), 256)
        np.save(
            os.path.join(
                outdir, 'audio/sushi-audio-{:05d}.npy'.format(i)), audio)
