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
#include "wavenet_infer.h"
#include <iostream>
#include <string>
#include <memory>
#include "nv_wavenet.cuh"
#include "matrix.h"

// Must match the wavenet channels
const int A = 256;
const int R = 64;
const int S = 256;
typedef nvWavenetInfer<float,float, R, S, A> MyWaveNet;

// ------------------------------------------------
// C-compatible function for wrapper
// ------------------------------------------------
void* wavenet_construct(int sample_count,
                        int batch_size,
                        float* embedding_prev,
                        float* embedding_curr,
                        int num_layers,
                        int max_dilation,
                        float** in_layer_weights_prev,
                        float** in_layer_weights_curr,
                        float** in_layer_biases,
                        float** res_layer_weights,
                        float** res_layer_biases,
                        float** skip_layer_weights,
                        float** skip_layer_biases,
                        float* conv_out_weight,
                        float* conv_end_weight,
                        int use_embed_tanh,
                        int implementation
                        ) {
    MyWaveNet* wavenet = new MyWaveNet(num_layers, max_dilation,
                                         batch_size, sample_count,
                                         implementation,
                                         use_embed_tanh);
    
    wavenet->setEmbeddings(embedding_prev, embedding_curr);

    for (int l = 0; l < num_layers; l++) {
        wavenet->setLayerWeights(l, in_layer_weights_prev[l],
                                    in_layer_weights_curr[l],
                                    in_layer_biases[l],
                                    res_layer_weights[l],
                                    res_layer_biases[l],
                                    skip_layer_weights[l],
                                    skip_layer_biases[l]);
    }

    // We didn't use biases on our outputs
    std::vector<float> dummy_bias_first(S, 0);
    std::vector<float> dummy_bias_second(A, 0);
    
    wavenet->setOutWeights(conv_out_weight, 
                           dummy_bias_first.data(),
                           conv_end_weight,
                           dummy_bias_second.data());

    return (void*)wavenet;
}

void wavenet_infer(void* wavenet,
                   int* samples,
                   float* cond_input,
                   int sample_count,
                   int batch_size) {
    assert(samples);
    Matrix outputSelectors(batch_size, sample_count);
    outputSelectors.randomize(0.5,1.0);

    MyWaveNet* myWaveNet = (MyWaveNet *)wavenet;
    myWaveNet->setInputs(cond_input, outputSelectors.data());

    int batch_size_per_block = ((batch_size % 4) == 0) ? 4 : ((batch_size % 2) == 0) ? 2 : 1;
    assert(myWaveNet->run(sample_count, batch_size, samples, batch_size_per_block, true));
    gpuErrChk(cudaDeviceSynchronize());
    return;
}

void wavenet_destruct(void* wavenet) {
    MyWaveNet* myWaveNet = (MyWaveNet *)wavenet;
    delete myWaveNet;
}

int get_R() {return R;}
int get_S() {return S;}
int get_A() {return A;}
