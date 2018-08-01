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

#include "nv_wavenet.cuh"
#include <cuda_profiler_api.h>
#include <stdio.h>
#include <vector>
#include <unistd.h>

template <typename T_weight, typename T_data, int R, int S, int A>
float getSampleRateT(int num_layers, int max_dilation, int batch_size, int batch_size_per_block, int num_samples, int num_samples_per_chunk, int mode) {

    // Set up initial activations


    int conditioning_size = num_samples * num_layers * batch_size * 2 * R * sizeof(float);
    float* conditioning = (float*)malloc(conditioning_size);

    if (conditioning == NULL) {
        fprintf(stderr, "\nERROR: Unable to allocate conditioning vectors.  Try running with fewer timesteps (-n)\n\n");        
        assert(false);
    }


    std::vector<float> randomSelector(batch_size*num_samples);
    for (int i=0; i<batch_size*num_samples; i++) {
        randomSelector[i] = (float) rand() / RAND_MAX;
    }

    float randomWeights[A*A];
    for (int i=0; i<R*R*16; i++) {
        randomWeights[i] = -0.5 + static_cast <float> (rand()) / static_cast <float> (RAND_MAX); 
    }

    nvWavenetInfer<T_weight, T_data, R, S, A> infer(num_layers, max_dilation, batch_size, num_samples, mode);
    for (int l=0; l<num_layers; l++) {
        infer.setLayerWeights(l, randomWeights, randomWeights, randomWeights, randomWeights, randomWeights, randomWeights, randomWeights);
    }
    infer.setOutWeights(randomWeights, randomWeights, randomWeights, randomWeights);
    infer.setInputs(conditioning, &randomSelector[0]); 
    gpuErrChk(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    gpuErrChk(cudaEventCreate(&start));
    gpuErrChk(cudaEventCreate(&stop));
    gpuErrChk(cudaEventRecord(start));
    int* mcYout;
    // because the chunked version copies repeatedly, we should measure it as well.
    gpuErrChk(cudaMallocHost(&mcYout, num_samples*batch_size*sizeof(int)));
    cudaProfilerStart();
    bool success = infer.run_chunks(num_samples_per_chunk, [](int*, int, int){}, num_samples, batch_size, mcYout, batch_size_per_block);
    gpuErrChk(cudaFreeHost(mcYout));

    gpuErrChk(cudaEventRecord(stop));

    gpuErrChk(cudaEventSynchronize(stop));
    float elapsed_time_ms;
    gpuErrChk(cudaEventElapsedTime(&elapsed_time_ms, start, stop));
    gpuErrChk(cudaDeviceSynchronize());

    free(conditioning);
    return success ? float(num_samples) / elapsed_time_ms : 0.f;

}

float getSampleRate(int precision, int r, int s, int a, int num_layers, int max_dilation, int batch_size, int batch_size_per_block, int num_samples, int num_samples_per_chunk, int mode) {
    assert(a==256);
    float sample_rate;
    if (r == 32) {
        assert(s==128);
        assert(a==256);
        if (precision == 16) {
	    sample_rate = getSampleRateT<half2,half,32,128,256>(num_layers, max_dilation, batch_size, batch_size_per_block, num_samples, num_samples_per_chunk, mode);
        }
        else {
            assert(precision==32);
	    sample_rate = getSampleRateT<float,float,32,128,256>(num_layers, max_dilation, batch_size, batch_size_per_block, num_samples, num_samples_per_chunk, mode);
        }
    }
    else {
        assert(r==64);
        if (precision == 16) {
            if (s==128) 
                sample_rate = getSampleRateT<half2,half,64,128,256>(num_layers, max_dilation, batch_size, batch_size_per_block, num_samples, num_samples_per_chunk, mode);
            else if (s==256)
                sample_rate = getSampleRateT<half2,half,64,256,256>(num_layers, max_dilation, batch_size, batch_size_per_block, num_samples, num_samples_per_chunk, mode);
            else
                assert(false);
        }
        else {
            assert(precision==32);
            if (s==128) 
                sample_rate = getSampleRateT<float,float,64,128,256>(num_layers, max_dilation, batch_size, batch_size_per_block, num_samples, num_samples_per_chunk, mode);
            else if (s==256)
                sample_rate = getSampleRateT<float,float,64,256,256>(num_layers, max_dilation, batch_size, batch_size_per_block, num_samples, num_samples_per_chunk, mode);
            else
                assert(false);
        }
    }
    return sample_rate;
}

std::vector<int> factorize(int v) {
    std::vector<int> rv;
    for (int i=1; i<=v; i++) {
        if ((v%i)==0) rv.push_back(i);
    }
    return rv;
}

std::vector<int> factorize_pow2(int v) {
    std::vector<int> rv;
    for (int i=1; i<=v; i <<= 1) {
        if ((v%i)==0) rv.push_back(i);
    }
    return rv;
}

int main(int argc, char* argv[]) {


    int num_layers = 20;
    int r = 64;
    int s = 128;
    int a = 256;
    int batch_size = 1;
    int batch_size_per_block = 1;
    int num_samples = 16384; 
    int max_dilation = 512;
    int mode = 0;
    int precision = 16;
    int num_samples_per_chunk = 2048;

    int c;
    while ((c = getopt (argc, argv, "l:r:s:a:b:n:c:d:m:p:t:")) != -1) {
        switch (c) {
            case 'l':
                num_layers = atoi(optarg);
                break;
            case 'r': 
                r = atoi(optarg);
                break;
            case 's':
                s = atoi(optarg);
                break;
            case 'a':
                a = atoi(optarg);
                break;
            case 'b':
                batch_size = atoi(optarg);
                break;
            case 'n':
                num_samples = atoi(optarg);
                break;
            case 'c':
                batch_size_per_block = atoi(optarg);
                break;
            case 'd':
                max_dilation = atoi(optarg);
                break;
            case 'm':
                mode = atoi(optarg);
                break;
            case 'p':
                precision = atoi(optarg);
                break;
            case 't':
                num_samples_per_chunk = atoi(optarg);
                break;
            default:
                assert(false);
        }
    }
    
    if (r != 32 && r != 64) {
        printf("ERROR: Only R=32,64 currently supported\n");
    }
    if (s != 128 && s != 256) {
        printf("ERROR: Only S=128 and S=256 currently supported\n");
    }
    if (a != 256) {
        printf("ERROR: Only A=256 currently supported\n");
    }


    printf("R: %d\n", r);
    printf("S: %d\n", s);
    printf("A: %d\n", a);
    printf("num layers: %d\n", num_layers);
    printf("max dilation: %d\n", max_dilation);
    printf("batch size: %d\n", batch_size);
    printf("batch size per block: %d\n", batch_size_per_block);
    printf("num samples: %d\n", num_samples);
    switch (mode) {
        case 0: printf("mode: AUTO\n"); break;
        case 1: printf("mode: SINGLE_block\n"); break;
        case 2: printf("mode: DUAL_block\n"); break;
        case 3: printf("mode: PERSISTENT\n"); break;
        default: assert(false);
    }
    assert(precision == 16 || precision == 32);
    printf("precision: fp%d\n", precision);

    srand(1);

    float sample_rate = getSampleRate(precision, r, s, a, num_layers, max_dilation, batch_size, batch_size_per_block, num_samples, num_samples_per_chunk, mode);
    printf("Sample rate: %f kHz\n", sample_rate);
}
