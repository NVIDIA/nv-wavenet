
"""
Generating pairs of mel-spectrograms and original audio
"""
import argparse
import json
import os
import random
import tensorflow as tf
import numpy as np
import sys

import wavenet_utils

# We're using the audio processing from Sushibot to make sure it matches
sys.path.insert(0, '../pytorch/sushibot')
from hparams import hparams
from utils.audio import melspectrogram


class Mel2SampOnehot(object):
    """
    This is the main class that calculates the spectrogram and returns the
    spectrogram, audio pair.
    """

    def __init__(self, training_files, segment_length, mu_quantization,
                 filter_length, hop_length, win_length, sampling_rate):
        audio_files = wavenet_utils.files_to_list(training_files)
        self.audio_files = audio_files
        random.seed(1234)
        random.shuffle(self.audio_files)
        self.segment_length = segment_length
        self.mu_quantization = mu_quantization
        self.sampling_rate = sampling_rate

    def get_mel(self, audio):
        audio_norm = audio / wavenet_utils.MAX_WAV_VALUE
        melspec = melspectrogram(audio_norm, hparams)
        melspec = melspec.transpose()
        return melspec

    def preprocess(self):
        mels = []
        targets = []
        for index in range(len(self.audio_files)):
            filename = self.audio_files[index]
            audio, sampling_rate = wavenet_utils.load_wav_to_torch(filename)
            if sampling_rate != self.sampling_rate:
                raise ValueError("{} {} SR doesn't match target {} SR".format(
                    sampling_rate, self.sampling_rate))

            audio = audio.astype(np.int32)
            if audio.shape[0] >= self.segment_length:
                max_audio_start = audio.shape[0] - self.segment_length
                audio_start = random.randint(0, max_audio_start)
                audio = audio[audio_start:audio_start+self.segment_length]
            else:
                audio = np.pad(
                    audio, (0, self.segment_length - audio.shape[0]), 'constant')

            mel = self.get_mel(audio)
            audio = wavenet_utils.mu_law_encode(
                audio / wavenet_utils.MAX_WAV_VALUE, self.mu_quantization)
            mels.append(mel)
            targets.append(audio)
        return (mels, targets)

    def __len__(self):
        return len(self.audio_files)


if __name__ == "__main__":
    """
    Turns audio files into mel-spectrogram representations for inference
    Uses the data portion of the config for audio processing parameters, 
    but ignores training files and segment lengths.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', "--audio_list", required=True, type=str,
                        help='File containing list of wavefiles')
    parser.add_argument('-o', "--output_dir", required=True, type=str,
                        help='Directory to put Mel-Spectrogram Tensors')
    parser.add_argument('-c', '--config', type=str,
                        help='JSON file for configuration')

    args = parser.parse_args()

    filepaths = wavenet_utils.files_to_list(args.audio_list)

    # Make directory if it doesn't exist
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
        os.chmod(args.output_dir, 0o775)

    # Parse config.  Only using data processing
    with open(args.config) as f:
        data = f.read()
    config = json.loads(data)
    data_config = config["data_config"]
    mel_factory = Mel2SampOnehot(**data_config)

    for filepath in filepaths:
        audio, sampling_rate = wavenet_utils.load_wav_to_torch(filepath)
        assert(sampling_rate == mel_factory.sampling_rate)
        mel = mel_factory.get_mel(audio)
        filename = os.path.basename(filepath)
        new_filepath = args.output_dir + '/' + filename + '.pt'
        print(new_filepath)
        np.save(mel, new_filepath)
