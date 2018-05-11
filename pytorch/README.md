# PyTorch Wrapper for NV-WaveNet

Allows NV-WaveNet to be called from PyTorch.  Currently tested on PyTorch 0.4

## Try It
1. Build nv-wavenet and C-wrapper: `cd pytorch; make`
2. Build the PyTorch extension: `python build.py`
3. Download a pre-trained WaveNet [here](https://drive.google.com/file/d/1TTR8oCdlQrM5gi7Y-rHrQ_v8kl8pd14o/view?usp=sharing) 
4. Download some inputs [here](https://drive.google.com/file/d/1_eNDHwvDc2r7RCxpbrPEWQkwFVs0rnfz/view?usp=sharing)
5. Run the inference: `python nv_wavenet_test.py`
This produces a file `audio.wav`.

The audio quality is due to the particular WaveNet and inputs, not the nv-wavenet implementation.  To check, you can compare your generated audio file to [this audio file](https://drive.google.com/file/d/1Xhd0VhGxyUgmb-QGNHW2tJILX1bro9d5/view?usp=sharing) generated from the same WaveNet and inputs but without nv-wavenet.  They sound the same, but the downloaded audio file took ~10 minutes to generate using a non-nv-wavenet implementation of WaveNet inference on a Nvidia V100.

## Usage
### 1. Choose number of channels for your network
NV-WaveNet is an implementation of the [Deep Voice](https://arxiv.org/abs/1702.07825) network architecture seen below:
![DeepVoice WaveNet](https://drive.google.com/file/d/1Zo-c5VzPLSEQlD_SyNoly3XWS0A7fi5s/view?usp=sharing)

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
These are the weights of the causal dilated convolutions.  It assumed the dilation starts at 1 and doubles each layer up to `max_dilation`.  NV-WaveNet assumes kernel size 2 in the causal dilated convolutions
 
 - `dilate_biases`:  List of tensors of size 2\***R** .
The biases of the dilated convolutions.

 - `max_dilation`: Integer
It's assumed the convolution dilation will double on each layer up to this value.  E.g. with 4 layers and a `max\_dilation`==2, dilations will be [1,2,1,2]

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

 - `implementation`:  An `nv_wavenet.Impl` Enum, which is one of: `AUTO`, `SINGLE_BLOCK`, `DUAL_BLOCK`, or `PERSISTENT`

NV-WaveNet focuses on the auto-regressive portion of the calculation, so there are no separate tensors in the NV-WaveNet for upsampling local or global features.  Upsampling calculations should be done separately giving one feature vector per layer per sample, which will be added to the activations before the non-linearities (as shown in figure 1).

`infer` returns an tensor of integers corresponding to the one-hot representation of the audio samples.
