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
"""
Tests that the NV-WaveNet class is producing audio
"""
import torch
from scipy.io.wavfile import write
import numpy as np
import nv_wavenet

MAX_WAV_VALUE = 32768.0

def mu_law_decode_numpy(x, mu_quantization=256):
    assert(np.max(x) <= mu_quantization)
    assert(np.min(x) >= 0)
    mu = mu_quantization - 1.
    # Map values back to [-1, 1].
    signal = 2 * (x / mu) - 1
    # Perform inverse of mu-law transformation.
    magnitude = (1 / mu) * ((1 + mu)**np.abs(signal) - 1)
    return np.sign(signal) * magnitude

if __name__ == '__main__':
    model = torch.load("model.pt")
    wavenet = nv_wavenet.NVWaveNet(**model)
    cond_input = torch.load("cond_input.pt")
    
    samples = wavenet.infer(cond_input, nv_wavenet.Impl.PERSISTENT)[0]
    
    audio = mu_law_decode_numpy(samples.cpu().numpy(), 256)
    audio = MAX_WAV_VALUE * audio
    wavdata = audio.astype('int16')
    write('audio.wav',16000, wavdata)
