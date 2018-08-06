# *****************************************************************************
#  Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
# 
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#      * Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#      * Neither the name of the NVIDIA CORPORATION nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
# 
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# 
# *****************************************************************************
import os
from scipy.io.wavfile import write
import torch
import nv_wavenet
import utils

def chunker(seq, size):
    """
    https://stackoverflow.com/a/434328
    """
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


def main(mel_files, model_filename, output_dir, batch_size, implementation, verbose):
    mel_files = utils.files_to_list(mel_files)
    model = torch.load(model_filename)['model']
    wavenet = nv_wavenet.NVWaveNet(**(model.export_weights()))
    
    for files in chunker(mel_files, batch_size):
        mels = []
        for file_path in files:
            print(file_path)
            mel = torch.load(file_path)
            mel = utils.to_gpu(mel)
            mels.append(torch.unsqueeze(mel, 0))

        upsamp_window = model.upsample.kernel_size[0]
        upsamp_stride = model.upsample.stride[0]
        padding = upsamp_window // (upsamp_stride * 2) + 1

        mels = torch.cat(mels, 0)
        length = mel.shape[-1]
        split_size = 80
        splits = range(0, length, split_size)
        if verbose:
            from tqdm import tqdm
            splits = tqdm(splits)

        audio_data = []
        for left in splits:
            right = min(left + split_size, length)
            padded_left = max(left - padding, 0)
            padded_right = min(right + padding, length)
            cond_input = model.get_cond_input(mels[:, :, padded_left:padded_right].cuda())

            cutoff_left = (left - padded_left) * upsamp_stride
            cutoff_right = (padded_right - right) * upsamp_stride
            cond_input = cond_input[:, :, :, cutoff_left:cond_input.shape[-1]-cutoff_right]

            synthesized = wavenet.infer(cond_input, implementation)
            audio_data.append(synthesized.cpu())
        audio_data = torch.cat(audio_data, -1)

        for i, file_path in enumerate(files):
            file_name = os.path.splitext(os.path.basename(file_path))[0]
            
            audio = utils.mu_law_decode_numpy(audio_data[i,:].cpu().numpy(), 256)
            audio = utils.MAX_WAV_VALUE * audio
            wavdata = audio.astype('int16')
            write("{}/{}.wav".format(output_dir, file_name),
                  16000, wavdata)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', "--filelist_path", required=True)
    parser.add_argument('-c', "--checkpoint_path", required=True)
    parser.add_argument('-o', "--output_dir", required=True)
    parser.add_argument('-b', "--batch_size", default=1)
    parser.add_argument('-i', "--implementation", type=str, default="persistent",
                        help="""Which implementation of NV-WaveNet to use.
                        Takes values of single, dual, or persistent""" )
    parser.add_argument('-v', "--verbose", action='store_true')
    
    args = parser.parse_args()
    if args.implementation == "auto":
        implementation = nv_wavenet.Impl.AUTO
    elif args.implementation == "single":
        implementation = nv_wavenet.Impl.SINGLE_BLOCK
    elif args.implementation == "dual":
        implementation = nv_wavenet.Impl.DUAL_BLOCK
    elif args.implementation == "persistent":
        implementation = nv_wavenet.Impl.PERSISTENT
    else:
        raise ValueError("implementation must be one of auto, single, dual, or persistent")
    
    main(args.filelist_path, args.checkpoint_path, args.output_dir, args.batch_size,
         implementation, args.verbose)
