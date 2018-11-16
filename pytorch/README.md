# PyTorch Implementation of NV-WaveNet
This directory now contains code for both the PyTorch Wrapper for the NV-WaveNet inference code, as well as PyTorch code for training a new WaveNet that translates mel-spectrograms to audio samples using the NV-WaveNet code at inference time.

First we cover the wrapper, which can be used with a pre-existing WaveNet for inference.  Then we cover training a new WaveNet below
 
# PyTorch Wrapper for NV-WaveNet

Allows NV-WaveNet to be called from PyTorch.  Currently tested on PyTorch 0.4

## Try It
1. Update the ``Makefile`` with the appropriate ``ARCH=sm_70``. Find your ARCH here: https://developer.nvidia.com/cuda-gpus. For example, NVIDIA Titan V has 7.0 compute capability; therefore, it's correct ``ARCH`` parameter is ``sm_70``.
2. Build nv-wavenet and C-wrapper: `cd pytorch; make`
3. Install the PyTorch extension: `python build.py install`
4. Download a pre-trained WaveNet [here](https://drive.google.com/file/d/1TTR8oCdlQrM5gi7Y-rHrQ_v8kl8pd14o/view?usp=sharing) 
5. Download some inputs [here](https://drive.google.com/file/d/1_eNDHwvDc2r7RCxpbrPEWQkwFVs0rnfz/view?usp=sharing)
6. Run the inference: `python nv_wavenet_test.py`
This produces a file `audio.wav`.

To check the results, you can compare your generated audio file to [this audio file](https://drive.google.com/file/d/1Xhd0VhGxyUgmb-QGNHW2tJILX1bro9d5/view?usp=sharing) generated from the same WaveNet and inputs but without nv-wavenet.  They sound the same, but the downloaded audio file took ~10 minutes to generate using a non-nv-wavenet implementation of WaveNet inference on a Nvidia V100.

## Usage
### 1. Choose number of channels for your network
NV-WaveNet is an implementation of the [Deep Voice](https://arxiv.org/abs/1702.07825) network architecture seen below:
![DeepVoice WaveNet](https://drive.google.com/uc?export=view&id=1Zo-c5VzPLSEQlD_SyNoly3XWS0A7fi5s)

NV-WaveNet has 3 parameters for number of channels in the convolutions:

 - **A** - number of channels in the output layers
Audio samples are discretized into **A** bins, which is also constrained to be the size of the final softmax layer, as well as the output layer of the final two convolutions.

 - **R** - The number of channels in residual layers/embedding.
In the network, the samples are represented by **R** dimensional embeddings, which constrains the number of outputs in the residual convolution as well.

 - **S** - The number of channels in the skip layers.
These parameters must be set at compile time.  Check the parent README for combinations of parameters that have been tested.  By default they are set to:
**A** = 256
**R** = 64
**S** = 256
To set these parameters edit them at the top of `wavenet_infer.cu`.  Once saved, rebuild the code with `make; python build.py`

### 2. The `NVWaveNet` class is initialized with the tensors of the WaveNet:
The main class is the `NVWaveNet` class in the `nv_wavenet` module.  It requires the following arguments during construction:
 - `embedding_prev`:  **A** x **R** tensor.
DeepVoice begins with a 2x1 causal convolution on the one-hot representation of the vectors.  This essentially means there is an embedding for the current and previous audio samples. This is the embedding matrix for the previous sample.
 
 - `embedding_curr`:  **A** x **R** tensor.
The current time embedding.
 
 - `conv_out_weight`:  **A** x **S** tensor.
DeepVoice ends with two convolutions coming from the skip connections (with no bias terms).  This is the weight matrix representing the first of those convolutions.
 
 - `conv_end_weight`: **A** x **A** tensor.
This is the second of the two output convolutions.
 
 - `dilate_weights`: List of tensors of size (2\***R**) x **R** x 2 .
These are the weights of the causal dilated convolutions.  It assumed the dilation starts at 1 and doubles each layer up to `max_dilation`.  NV-WaveNet assumes kernel size 2 in the causal dilated convolutions.  The first **R** channels are those going through the `tanh` nonlinearity and the second **R** channels are those going through the sigmoid nonlinearity.
 
 - `dilate_biases`:  List of tensors of size 2\***R** .
The biases of the dilated convolutions.

 - `max_dilation`: Integer
It's assumed the convolution dilation will double on each layer up to this value.  E.g. with 4 layers and a `max_dilation==2`, dilations will be [1,2,1,2]

 - `res_weights`: List of tensors of size **R** x **R** .
The convolutions for residual connections.

 - `res_biases`: List of tensors of size **R** .
The biases for the residual connections.

 - `skip_weights`: List of tensors of size **S** x **R**
The convolutions for the skip connections
 
 - `skip_biases`: List of tensor
s of size **S**
 The biases for the skip connections
 
 - `use_embed_tanh`: Boolean
In the DeepVoice implementation, there is a `tanh` non-linearity after an initial 2x1 convolution of the audio embeddings.  However, this non-linearity is not in the original WaveNet description.  In order to make NV-WaveNet more compatible with other WaveNet implementations the `tanh` can be removed by setting this parameter to false.  Then if desired the initial convolution can be removed by setting the `embedding_prev` to all zeros.

### 3. The classes `infer` function is called with the inputs the WaveNet is to be conditioned on:
The constructed NVWaveNet instance has an `infer` method that takes a tensor of conditional activations.  `infer` takes two arguments:
 - `cond_input`:  A tensor of size  (2\***R**) x (batch size) x (# of layers) x (# of samples)
The first **R** channels are those going through the `tanh` nonlinearity and the second **R** channels are those going through the sigmoid nonlinearity.

 - `implementation`:  An `nv_wavenet.Impl` Enum, which is one of: `AUTO`, `SINGLE_BLOCK`, `DUAL_BLOCK`, or `PERSISTENT`

NV-WaveNet focuses on the auto-regressive portion of the calculation, so there are no separate tensors in the NV-WaveNet for upsampling local or global features.  Upsampling calculations should be done separately giving one feature vector per layer per sample, which will be added to the activations before the non-linearities (as shown in figure 1).

`infer` returns an tensor of integers corresponding to the one-hot representation of the audio samples.


# Training and Inference with NV-WaveNet
This descibes the code for training a new WaveNet that translates mel-spectrograms of audio to audio samples using the NV-WaveNet code for inference.

Currently tested on PyTorch 0.4, and 16khz audio

## Try It
1. First build the nv-wavenet wrapper described above.  If you're using the default network size in `config.json` then you should just need `make; python build.py`.
2. Download some data.  Here we're using CMU's Arctic 0.95, speaker id `clb`:
`mkdir data;cd data`
`wget -A ".wav" -nd -r http://festvox.org/cmu_arctic/cmu_arctic/cmu_us_clb_arctic/wav`
`cd ..`
3. Make a list of the file names to use for training/testing
`ls data/*.wav | tail -n+10 > train_files.txt` 
`ls data/*.wav | head -n10 > test_files.txt` 
4. Train your WaveNet:
`mkdir checkpoints`
`python train.py -c config.json`
For multi-GPU training replace `train.py` with `distributed.py`.  Only tested with single node and NCCL.
Audio is audible at 10k iterations, we typically use closer to 300k iterations.
5. Make test set mel-spectrograms
`python mel2samp_onehot.py -a test_files.txt -o . -c config.json`
6. Do inference with your network
`ls *.pt > mel_files.txt`
`python inference.py -f mel_files.txt -c checkpoints/wavenet_10000 -o .`

You should now have your test wavfiles in your directory.
