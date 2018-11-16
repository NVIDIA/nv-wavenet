/******************************************************************************
 * Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/
#include <vector>
#include <THC/THC.h>
#include <torch/torch.h>
#include "wavenet_infer.h"

int infer(at::Tensor samples_tensor,
          int sample_count,
          int batch_size,
          at::Tensor embed_prev_tensor,
          at::Tensor embed_curr_tensor,
          at::Tensor conv_out_tensor,
          at::Tensor conv_end_tensor,
          at::Tensor cond_input_tensor,
          int num_layers,
          int use_embed_tanh,
          int max_dilation,
          int implementation,
          std::vector<at::Tensor>& layers) {
    int* samples = samples_tensor.data<int>();

    float* embedding_prev = embed_prev_tensor.data<float>();
    float* embedding_curr = embed_curr_tensor.data<float>();
    float* conv_out = conv_out_tensor.data<float>();
    float* conv_end = conv_end_tensor.data<float>();
    float* cond_input = cond_input_tensor.data<float>();

    float** in_layer_weights_prev = (float**)malloc(num_layers*sizeof(float*));
    float** in_layer_weights_curr = (float**)malloc(num_layers*sizeof(float*));
    float** in_layer_biases = (float**)malloc(num_layers*sizeof(float*));
    float** res_layer_weights = (float**)malloc(num_layers*sizeof(float*));
    float** res_layer_biases = (float**)malloc(num_layers*sizeof(float*));
    float** skip_layer_weights = (float**)malloc(num_layers*sizeof(float*));
    float** skip_layer_biases = (float**)malloc(num_layers*sizeof(float*));
    for (int i=0; i < num_layers; i++) {
        int idx = i*7;
	in_layer_weights_prev[i] = layers[idx].data<float>();
	in_layer_weights_curr[i] = layers[idx+1].data<float>();
	in_layer_biases[i] = layers[idx+2].data<float>();
	res_layer_weights[i] = layers[idx+3].data<float>();
	res_layer_biases[i] = layers[idx+4].data<float>();
	skip_layer_weights[i] = layers[idx+5].data<float>();
	skip_layer_biases[i] = layers[idx+6].data<float>();
    }

    wavenet_infer(sample_count,
                  batch_size,
                  embedding_prev,
                  embedding_curr,
                  num_layers,
                  max_dilation,
                  in_layer_weights_prev,
                  in_layer_weights_curr,
                  in_layer_biases,
                  res_layer_weights,
                  res_layer_biases,
                  skip_layer_weights,
                  skip_layer_biases,
                  conv_out,
                  conv_end,
                  use_embed_tanh,
                  cond_input,
                  implementation,
                  samples);

    free(in_layer_weights_prev);
    free(in_layer_weights_curr);
    free(in_layer_biases);
    free(res_layer_weights);
    free(res_layer_biases);
    free(skip_layer_weights);
    free(skip_layer_biases);
    return 1;
}

int num_res_channels(void){return get_R();}
int num_skip_channels(void){return get_S();}
int num_out_channels(void){return get_A();}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("infer", &infer, "NV-WaveNet inference");
    m.def("num_res_channels", &num_res_channels, "");
    m.def("num_skip_channels", &num_skip_channels, "");
    m.def("num_out_channels", &num_out_channels, "");
}
