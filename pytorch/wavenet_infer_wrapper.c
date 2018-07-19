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
#include <stdarg.h>
#include <THC/THC.h>
#include "wavenet_infer.h"
extern THCState *state;

int infer(THCudaIntTensor* samples_tensor,
          int sample_count,
          int batch_size,
          THCudaTensor* embed_prev_tensor,
          THCudaTensor* embed_curr_tensor,
          THCudaTensor* conv_out_tensor,
          THCudaTensor* conv_end_tensor,
          THCudaTensor* cond_input_tensor,
          int num_layers,
          int use_embed_tanh,
          int max_dilation,
          int implementation, ...) {
    int* samples = THCudaIntTensor_data(state, samples_tensor);
      
    float* embedding_prev = THCudaTensor_data(state, embed_prev_tensor);
    float* embedding_curr = THCudaTensor_data(state, embed_curr_tensor);
    float* conv_out = THCudaTensor_data(state, conv_out_tensor);
    float* conv_end = THCudaTensor_data(state, conv_end_tensor);
    float* cond_input = THCudaTensor_data(state, cond_input_tensor);

    float** in_layer_weights_prev = malloc(num_layers*sizeof(float*));
    float** in_layer_weights_curr = malloc(num_layers*sizeof(float*));
    float** in_layer_biases = malloc(num_layers*sizeof(float*));
    float** res_layer_weights = malloc(num_layers*sizeof(float*));
    float** res_layer_biases = malloc(num_layers*sizeof(float*));
    float** skip_layer_weights = malloc(num_layers*sizeof(float*));
    float** skip_layer_biases = malloc(num_layers*sizeof(float*));
    va_list layers;
    va_start(layers, implementation);
    for (int i=0; i < num_layers; i++) {
        in_layer_weights_prev[i] = THCudaTensor_data(state, va_arg(layers, THCudaTensor*));
        in_layer_weights_curr[i] = THCudaTensor_data(state, va_arg(layers, THCudaTensor*));
        in_layer_biases[i] = THCudaTensor_data(state, va_arg(layers, THCudaTensor*));
        res_layer_weights[i] = THCudaTensor_data(state, va_arg(layers, THCudaTensor*));
        res_layer_biases[i] = THCudaTensor_data(state, va_arg(layers, THCudaTensor*));
        skip_layer_weights[i] = THCudaTensor_data(state, va_arg(layers, THCudaTensor*));
        skip_layer_biases[i] = THCudaTensor_data(state, va_arg(layers, THCudaTensor*));
    }				
    va_end(layers);
	  
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
