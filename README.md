# Introduction

nv-wavenet is a CUDA reference implementation of autoregressive [WaveNet](https://arxiv.org/abs/1609.03499) inference.  In particular, it implements the WaveNet variant described by [Deep Voice](https://arxiv.org/abs/1702.07825).  nv-wavenet only implements the autoregressive portion of the network; conditioning vectors must be provided externally. More details about the implementation and performance can be found on the [NVIDIA Developer Blog](https://devblogs.nvidia.com/nv-wavenet-gpu-speech-synthesis/).

Channel counts are provided as template parameters.  The following channel count combinations have been tested and are expected to function correctly:

* 32 residual channels, 128 skip channels, 256 audio channels
* 64 residual channels, 128 skip channels, 256 audio channels
* 64 residual channels, 256 skip channels, 256 audio channels

The implementation provides three different variants, with different complexity, sample rate, throughput and resource characteristics:

* Single-Block: implements the entire network in a single thread block. Each thread block must read all model weights per sample, and thus sample rate is limited by the rate at which a single Streaming Multiprocessor can read weights. 
* Dual-Block: implements the network across two collaborating thread blocks. As these blocks may now span multiple Streaming Multiprocessors, this implementation can support a larger model at a given sample rate.
* Persistent: loads all weights into the register file, where they persist for the entire inference.  

In all three implementations, a single kernel runs inference for potentially many samples.

# Usage

`nv_wavenet.cuh` provides a templated class `nvWavenetInfer`.  The template parameters are:
* `T_weight` : should be `float` for fp32 inference, `half2` for fp16 inference
* `T_data` : should be `float` for fp32 inference, `half` for fp16 inference
* `R` : the number of residual channels
* `S` : the number of skip channels
* `A` : the number of audio channels

The `nvWavenetInfer` constructor accepts the following arguments:
* `numLayers` : the number of residual layers in the WaveNet
* `maxDilation` : the maximum dilation amount.  The dilated convolution of each residual layer will have dilation equal to twice the dilation of the prior layer, until this maximum value is reached.  The next layer will then reset its dilation to 1.
* `batchSize` : the inference batch size (the number of utterances to generate in parallel)
* `sampleCount` : the number of audio samples to generate
* `implementation` : the implementation variant to use, as defined by the `nvWavenetInfer::Implementation` enum.  Options are `SINGLE_BLOCK`, `DUAL_BLOCK` and `PERSISTENT`
* `tanhEmbed` : specifies whether the result of the input embedding should pass through a tanh

Once the `nvWavenetInfer` object is constructed, it is necessary to upload weights for the model.  Weight matrices are provided as `float*` arrays, in column-major order.  In the fp16 case, data conversion and vectorization is provided automatically by the weight upload functions. The provided pointers can be on the host or on the device - in either case, the data will be copied to a buffer belonging to the `NvWavenetInfer` object.

`nvWavenetInfer::setEmbeddings()` uploads the embedding table for the causal input.
`nvWavenetInfer::setLayerWeights()` uploads all necessary weights for a single residual layer.
`nvWavenetInfer::setOutWeights()` uploads all weights for the final output layers prior to the softmax.

The `nvWavenetInfer::setInputs()` method allows the user to upload conditioning vectors and random values for use by the random sampling post-softmax.  While setInputs does accept device pointers, it will still copy/convert the data into the `NvWavenetInfer` object's allocation. For efficient deployment where the conditioning vectors / random values are already present in GPU memory, this method should be modified to simply update the necessary pointers.

# Testing

nv-wavenet includes a simple reference implementation in `nv_wavenet_reference.h` and `nv_wavenet_reference.cpp`.  `nv_wavenet_test.cu` runs the reference implementation against the CUDA configuration for several configurations with random weights.  To run:
```
make nv_wavenet_test
./nv_wavenet_test
```

# Performance

`nv_wavenet_perf.cu` provides a simple performance test.

Before performance testing, it is recommended to fix the GPU clocks using `nvidia-smi`.  To query available clocks, run `nvidia-smi -q -d SUPPORTED_CLOCKS`.  The clock can then be set using `nvidia-smi -ac`

To build and run the performance test, run:
```
make nv_wavenet_perf

./nv_wavenet_perf <-l num_layers> <-r residual__channels> <-s skip_channels> <-a audio_channels> <-b batch_size> <-c batch_size_per_block> <-n num_samples> <-d max_dilation> <-m mode> <-p precision>
```
Finding the best performance at a particular sample rate will require experimenting with different values for `batch_size`, `batch_size_per_block` and mode.  `batch_size` must be a multiple of `batch_size_per_block`

# Open Source License

nv-wavenet is released by NVIDIA Corporation under the "New BSD" open-source license:

```
Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
   *  Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
   *  Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
   *  Neither the name of the NVIDIA CORPORATION nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```
