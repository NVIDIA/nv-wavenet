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

#include "cuda_fp16.h"
#include <stdint.h>
#include <stdio.h>
#include <assert.h>
#include <vector>
#include <algorithm>

#include "matrix_math.cuh"
#include "softmax.cuh"
#include "nv_wavenet_util.cuh"
#include "nv_wavenet_conversions.cuh"

template <typename T_weight, typename T_data >
struct nv_wavenet_params {
    int num_samples;
    int num_samples_per_chunk;
    int blocks_per_sample;
    int init_sample;
    int batch_size;
    int num_layers;
    int* yInPrev;
    int* yInCur;
    T_data* embedPrev;
    T_data* embedCur;
    bool tanhEmbed;
    T_weight* Wprev;
    T_data* L;
    T_weight* Wcur;
    T_data* B;
    T_weight* Wres;
    T_data* Bres;
    T_weight* Wskip;
    T_data* Bskip;
    T_data* xt;
    T_data* xtmd;
    T_data* xtOut;
    T_data* a_prev;
    T_data* skip_in;
    T_data* skip_out;
    T_weight* WskipOut;
    T_data* BskipOut;
    T_data* skipOutFinal;
    T_data* skipOutAccumulate;
    T_weight* Wout;
    T_data* Bout;
    T_data* out;
    T_data* outAccumulate;
    T_data* p;
    float* outputSelectors;
    int* yOut;
    bool dumpActivations;
    int maxDilation;

    T_data* h;
    volatile int*    hSample;
    volatile int*    ySample;

};

template <typename T_weight, typename T_data, int R, int BATCH_UNROLL>
__device__ void nv_wavenet_prev(int sample, int thread_id, int num_layers, int maxDilation, int batch_offset, int batch_size, T_weight* Wprev, T_data* L, T_data* xt, T_data* a_prev, bool dumpActivations) {
    const int WV = sizeof(T_weight)/sizeof(T_data);
    T_weight weights[R/WV];

    T_data accum[BATCH_UNROLL];

    __shared__ T_data xtmd_sh[2][BATCH_UNROLL][R];

    if (thread_id < 2*R) {
        int row = thread_id;
        int ping_pong = 0;
        int dilation = 1;
        for (int layer=0; layer<num_layers; layer++) {
            int sample_offset = (sample - dilation) % (maxDilation+1);
            T_data* xtmd = xt + sample_offset*(num_layers+1)*R*batch_size;
            if (row < R) {
    #pragma unroll
                for (int b=0; b<BATCH_UNROLL; b++) {
                    xtmd_sh[ping_pong][b][row] = (dilation <= sample) ? xtmd[layer*batch_size*R + (batch_offset+b)*R + row] : (T_data)0.f;
                }
            }
            ping_pong = ping_pong ^ 1;
            dilation = dilation << 1;
            if (dilation > maxDilation) dilation = 1;
            namedBarrierSync(3,4*R);
        }
    }
    else if (thread_id< 4*R) {
        int ping_pong = 0;
        int row = thread_id - 2*R;
        for (int layer=0; layer<num_layers; layer++) {
            loadWeights<2*R,R>(weights,Wprev,layer,row);
            namedBarrierSync(3,4*R);
            GEMM<R,4,BATCH_UNROLL>(weights,xtmd_sh[ping_pong],accum);
            for (int b=0; b<BATCH_UNROLL; b++) { 
                a_prev[layer*batch_size*2*R + (batch_offset+b)*2*R + row] = accum[b];
            }
            ping_pong = ping_pong ^ 1;
        }
    }

}

template <typename T_weight, typename T_data, int R, int BATCH_UNROLL>
__device__ void nv_wavenet_cur(int sample, int row, int num_layers, int batch_offset, int batch_size, T_weight* Wcur, T_data* B, T_data* L, T_data xt_sh[BATCH_UNROLL][R], T_data a_cur_sh[BATCH_UNROLL][2*R], T_data* a_prev){
    const int WV = sizeof(T_weight)/sizeof(T_data);
    T_weight weights[R/WV];
    T_data accum[BATCH_UNROLL];
    T_data bias;
    namedBarrierSync(1,3*R);
    for (int layer=0; layer<num_layers; layer++) {
        loadWeights<2*R,R>(weights,Wcur,layer,row);
        bias = B[layer*2*R+row];
        T_data conditioning[BATCH_UNROLL];
        T_data a_prev_reg[BATCH_UNROLL];
        for (int b=0; b<BATCH_UNROLL; b++) {
            conditioning[b] = L[sample*num_layers*batch_size*2*R + layer*batch_size*2*R + (batch_offset+b)*2*R + row];
            a_prev_reg[b]= a_prev[layer*batch_size*2*R + (batch_offset+b)*2*R + row];
        }
        __syncthreads();
        GEMM<R,4,BATCH_UNROLL>(weights,xt_sh,accum);
        for (int b=0; b<BATCH_UNROLL; b++) { 
            accum[b] += a_prev_reg[b];
            accum[b] += bias; 
            accum[b] += conditioning[b];
            a_cur_sh[b][row] = (row < R) ? _tanh(accum[b]) : sigmoid(accum[b]);
        }
        namedBarrierSync(1,3*R);
    }
}

template <typename T_weight, typename T_data, int R, int S, int BATCH_UNROLL, bool DUAL_BLOCK=false>
__device__ void nv_wavenet_pointwise(int sample, int row, int num_layers, int batch_offset, int batch_size, T_data* xtmd, T_data xt_sh[BATCH_UNROLL][R], T_data a_cur_sh[BATCH_UNROLL][2*R], T_data h_sh[BATCH_UNROLL][R], T_data* h, volatile int* hSample) {
    namedBarrierSync(1,3*R);
    for (int layer=0; layer<num_layers; layer++) {
        __syncthreads(); 
        namedBarrierSync(1,3*R);
        for (int b=0; b<BATCH_UNROLL; b++) {
            T_data val_lo = a_cur_sh[b][row];
            T_data val_hi = a_cur_sh[b][row + R];
            T_data val = val_lo * val_hi;
            h_sh[b][row] = val;
            if (DUAL_BLOCK) h[layer*batch_size*R + (batch_offset+b)*R + row] = val;
        }
        if (DUAL_BLOCK) {
            namedBarrierSync(2,2*R);
            __threadfence();
            if (row < BATCH_UNROLL) {
                hSample[layer*batch_size + batch_offset + row] = sample+1;
            }
        }
        else {
            namedBarrierSync(2,2*R+S);
        }
    }
}

template <typename T_weight, typename T_data, int R, int S, int BATCH_UNROLL, bool DUAL_BLOCK=false>
__device__ void nv_wavenet_res(int sample, int row, int num_layers, int maxDilation, int batch_offset, int batch_size, T_weight* Wres, T_data* Bres, T_data h_sh[BATCH_UNROLL][R], T_data xt_sh[BATCH_UNROLL][R], T_data* xt, T_data* xtOut, bool dumpActivations) {
    const int WV = sizeof(T_weight)/sizeof(T_data);
    T_weight weights[R/WV];
    T_data bias;
    T_data accum[BATCH_UNROLL];

    for (int layer=0; layer<num_layers; layer++) {
        __syncthreads();
        loadWeights<R,R>(weights,Wres,layer,row);
        namedBarrierSync(2,DUAL_BLOCK ? 2*R : 2*R+S);
        bias = Bres[layer*R+row];
        GEMM<R,2,BATCH_UNROLL>(weights,h_sh,accum);
        T_data* Xt = xt + (sample%(maxDilation+1))*(num_layers+1)*R*batch_size;
        for (int b=0; b<BATCH_UNROLL; b++) { 
            accum[b] += bias; 
            accum[b] += xt_sh[b][row];
            xt_sh[b][row] = accum[b];
            Xt[(layer+1)*batch_size*R + (batch_offset+b)*R + row] = accum[b];
            if (dumpActivations) xtOut[layer*batch_size*R + (batch_offset+b)*R + row] = accum[b];
        }
    }
}

#include "nv_wavenet_singleblock.cuh"
#include "nv_wavenet_dualblock.cuh"
#include "nv_wavenet_persistent.cuh"

__global__ void silenceInputs(int* yInPrev, int* yInCur, int size) {
    for (int i=threadIdx.x; i<size; i += blockDim.x) {
        yInPrev[i] = 128;
        yInCur[i] = 128;
    }
}

template <typename T_weight, typename T_data, int R=64, int S=128, int A=256>
class nvWavenetInfer {
    public:
    enum Implementation {
        AUTO = 0,
        SINGLE_BLOCK,
        DUAL_BLOCK,
        PERSISTENT,
        MANYBLOCK_NONPERSISTENT
    };
    protected:

        Implementation m_implementation;

        int m_numLayers;
        int m_maxBatch; 

        int* m_yOut;
        float* m_outputSelectors;

        T_data* m_embedPrev;
        T_data* m_embedCur;
        bool m_tanhEmbed;

        T_weight* m_Wprev;
        T_weight* m_Wcur;
        T_weight* m_Wres;
        T_weight* m_Wskip;

        T_data* m_Bh;
        T_data* m_Lh;
        T_data* m_Bres;
        T_data* m_Bskip;

        T_data* m_XtIn;
        T_data* m_hOut;
        T_data* m_aPrev;
        T_data* m_skipIn;
        T_data* m_skipOutFinalAccumulate;
        T_data* m_outAccumulate;
        int* m_yInPrev;
        int* m_yInCur;

        T_data* m_XtOut;
        T_data* m_skipOut;

        T_weight* m_WskipOut;
        T_data*   m_BskipOut;
        T_weight* m_Wout;
        T_data*   m_Bout;

        T_data* m_skipOutFinal;
        T_data* m_out;
        T_data* m_p;

        // For dual-block
        T_data* m_h;
        int*    m_hSample;
        int*    m_ySample;

        int m_maxDilation;

        int m_maxSamples;
        int m_num_samples_per_chunk;

        void setActivation(float* dst, float* src, size_t size) {
            gpuErrChk(cudaMemcpy(dst, src, size*sizeof(float), cudaMemcpyDefault));
        }
        void setActivation(half* dst, half* src, size_t size) {
            gpuErrChk(cudaMemcpy(dst, src, size*sizeof(half), cudaMemcpyDefault));
        }
        void setActivation(half* dst, float* src, size_t size) {
            convert_float2half(dst, src, size);
        }
        void getActivation(float* dst, float* src, size_t size) {
            gpuErrChk(cudaMemcpy(dst, src, size*sizeof(float), cudaMemcpyDefault));
        }
        void getActivation(float* dst, half* src, size_t size) {
            convert_half2float(dst, src, size);
        }
        void setLayerWeight(int layer, float* dst, float* src, int M, int K) {
            gpuErrChk(cudaMemcpy(dst + layer*M*K, src, M*K*sizeof(float), cudaMemcpyDefault));
        }
        void setLayerWeight(int layer, half2* dst, float* src, int M, int K) {
            convert_float2half2_vectorized(dst + layer*M*K/2, src, M, K);
        }
        void setLayerBias(int layer, float* dst, float* src, int M){
            gpuErrChk(cudaMemcpy(dst + layer*M, src, M*sizeof(float), cudaMemcpyDefault));
        }
        void setLayerBias(int layer, half* dst, float* src, int M){
            convert_float2half(dst + layer*M, src, M);
        }

    public:
        nvWavenetInfer (int numLayers, int maxDilation, int batchSize, int numSamples, int impl=0, bool tanhEmbed=true) : m_numLayers(numLayers), m_maxBatch(batchSize), m_maxSamples(numSamples), m_implementation((nvWavenetInfer::Implementation)impl), m_tanhEmbed(tanhEmbed) {

            m_num_samples_per_chunk = 0;
            m_maxDilation = maxDilation;

            gpuErrChk(cudaMalloc(&m_yOut, numSamples*batchSize*sizeof(int))); // one-hot vector represented as single value indicating which value is set
            gpuErrChk(cudaMemset(m_yOut, 0, numSamples*batchSize*sizeof(int))); 
            gpuErrChk(cudaMalloc(&m_outputSelectors, numSamples*batchSize*sizeof(float))); 

            gpuErrChk(cudaMalloc(&m_embedPrev, A*R*sizeof(T_data)));
            gpuErrChk(cudaMalloc(&m_embedCur, A*R*sizeof(T_data)));

            gpuErrChk(cudaMalloc(&m_Wprev, numLayers*2*R*R*sizeof(T_weight)));
            gpuErrChk(cudaMalloc(&m_Wcur, numLayers*2*R*R*sizeof(T_data)));
            gpuErrChk(cudaMalloc(&m_Bh, numLayers*2*R*sizeof(T_data)));
            gpuErrChk(cudaMalloc(&m_Lh, numSamples*numLayers*batchSize*2*R*sizeof(T_data)));
            gpuErrChk(cudaMalloc(&m_Wres, numLayers*R*R*sizeof(T_data)));
            gpuErrChk(cudaMalloc(&m_Bres, numLayers*R*sizeof(T_data)));
            gpuErrChk(cudaMalloc(&m_Wskip, numLayers*S*R*sizeof(T_data)));
            gpuErrChk(cudaMalloc(&m_Bskip, numLayers*S*sizeof(T_data)));
            gpuErrChk(cudaMalloc(&m_XtOut, numLayers*R*batchSize*sizeof(T_data)));
            gpuErrChk(cudaMalloc(&m_skipOut, numLayers*S*batchSize*sizeof(T_data)));

            // For now, just burn memory as though all layers had the maximum dilation value
            gpuErrChk(cudaMalloc(&m_XtIn, (m_maxDilation+1)*(numLayers+1)*R*batchSize*sizeof(T_data)));
            gpuErrChk(cudaMalloc(&m_hOut, numLayers*batchSize*R*sizeof(T_data)));
            gpuErrChk(cudaMalloc(&m_aPrev, numLayers*batchSize*2*R*sizeof(T_data)));
            gpuErrChk(cudaMalloc(&m_skipIn, numLayers*S*batchSize*sizeof(T_data)));
            gpuErrChk(cudaMalloc(&m_skipOutFinalAccumulate, A*batchSize*S/R*sizeof(T_data)));
            gpuErrChk(cudaMalloc(&m_outAccumulate, A*batchSize*A/R*sizeof(T_data)));
            gpuErrChk(cudaMalloc(&m_yInPrev, batchSize*sizeof(int))); // one-hot vector represented as single value indicating which value is set
            gpuErrChk(cudaMalloc(&m_yInCur, batchSize*sizeof(int))); // one-hot vector represented as single value indicating which value is set

            gpuErrChk(cudaMalloc(&m_WskipOut, A*S*sizeof(T_data)));
            gpuErrChk(cudaMalloc(&m_BskipOut, A*sizeof(T_data)));
            gpuErrChk(cudaMalloc(&m_Wout, A*A*sizeof(T_data)));
            gpuErrChk(cudaMalloc(&m_Bout, A*sizeof(T_data)));
            gpuErrChk(cudaMalloc(&m_skipOutFinal, A*batchSize*S/R*sizeof(T_data)));
            gpuErrChk(cudaMalloc(&m_out, A*batchSize*A/R*sizeof(T_data)));
            gpuErrChk(cudaMalloc(&m_p, A*batchSize*sizeof(T_data)));

            gpuErrChk(cudaMalloc(&m_h, numLayers*batchSize*R*sizeof(T_data)));
            gpuErrChk(cudaMalloc(&m_hSample, numLayers*batchSize*sizeof(int)));
            gpuErrChk(cudaMalloc(&m_ySample, batchSize*sizeof(int)));

            if (impl == PERSISTENT) {
                gpuErrChk(cudaMalloc(&m_skipOutFinalAccumulate, A*batchSize*S/R*sizeof(T_data))); 
                gpuErrChk(cudaMalloc(&m_outAccumulate, A*batchSize*A/R*sizeof(T_data))); 
            }

        }
        ~nvWavenetInfer() {
            gpuErrChk(cudaFree(m_yOut));
            gpuErrChk(cudaFree(m_outputSelectors));
            gpuErrChk(cudaFree(m_embedPrev));
            gpuErrChk(cudaFree(m_embedCur));
            gpuErrChk(cudaFree(m_Wprev));
            gpuErrChk(cudaFree(m_Wcur));
            gpuErrChk(cudaFree(m_Bh));
            gpuErrChk(cudaFree(m_Lh));
            gpuErrChk(cudaFree(m_Wres));
            gpuErrChk(cudaFree(m_Bres));
            gpuErrChk(cudaFree(m_Wskip));
            gpuErrChk(cudaFree(m_Bskip));
            gpuErrChk(cudaFree(m_XtOut));
            gpuErrChk(cudaFree(m_skipOut));
            gpuErrChk(cudaFree(m_XtIn));
            gpuErrChk(cudaFree(m_hOut));
            gpuErrChk(cudaFree(m_aPrev));
            gpuErrChk(cudaFree(m_skipIn));
            gpuErrChk(cudaFree(m_yInPrev));
            gpuErrChk(cudaFree(m_yInCur));
            gpuErrChk(cudaFree(m_WskipOut));
            gpuErrChk(cudaFree(m_BskipOut));
            gpuErrChk(cudaFree(m_Wout));
            gpuErrChk(cudaFree(m_Bout));
            gpuErrChk(cudaFree(m_skipOutFinal));
            gpuErrChk(cudaFree(m_out));
            gpuErrChk(cudaFree(m_p));

            if (m_implementation == PERSISTENT) {
                gpuErrChk(cudaFree(m_skipOutFinalAccumulate));
                gpuErrChk(cudaFree(m_outAccumulate));
            }
        }
        virtual void setEmbeddings (float* embedPrev, float* embedCur) {
            setActivation(m_embedPrev, embedPrev, A*R);
            setActivation(m_embedCur, embedCur, A*R);
        }
        virtual void setLayerWeights (int layer, float* Wprev, float* Wcur, float* Bh, float* Wres, float* Bres, float* Wskip, float* Bskip) {
            setLayerWeight(layer, m_Wprev, Wprev, 2*R, R);
            setLayerWeight(layer, m_Wcur, Wcur, 2*R, R);
            setLayerWeight(layer, m_Wres, Wres, R, R);
            setLayerWeight(layer, m_Wskip, Wskip, S, R);

            setLayerBias(layer, m_Bh, Bh, 2*R);
            setLayerBias(layer, m_Bres, Bres, R);
            setLayerBias(layer, m_Bskip, Bskip, S);
        }
        virtual void setOutWeights (float* Wzs, float* Bzs, float* Wza, float* Bza) {
            setLayerWeight(0, m_WskipOut, Wzs, A, S);
            setLayerBias(0, m_BskipOut, Bzs, A);
            setLayerWeight(0, m_Wout, Wza, A, A);
            setLayerBias(0, m_Bout, Bza, A);
        }

        void setInputs (float* Lh, float* outputSelectors) {
            silenceInputs<<<1,256>>>(m_yInPrev, m_yInCur, m_maxBatch);
            setActivation(m_Lh, Lh, m_maxSamples*m_numLayers*m_maxBatch*2*R);
            gpuErrChk(cudaMemcpy(m_outputSelectors, outputSelectors, m_maxSamples*m_maxBatch*sizeof(float), cudaMemcpyHostToDevice));

        }
        void setInputs (half* Lh, float* outputSelectors) {
            silenceInputs<<<1,256>>>(m_yInPrev, m_yInCur, m_maxBatch);
            setActivation(m_Lh, Lh, m_maxSamples*m_numLayers*m_maxBatch*2*R);
            gpuErrChk(cudaMemcpy(m_outputSelectors, outputSelectors, m_maxSamples*m_maxBatch*sizeof(float), cudaMemcpyHostToDevice));

        }

        void getXtOut(int layer, float* hXt) { getActivation(hXt, m_XtOut + layer*m_maxBatch*R, m_maxBatch*R); }
        void getSkipOut(int layer, float* hSkipOut) { getActivation(hSkipOut, m_skipOut + layer*m_maxBatch*S, m_maxBatch*S); }
        void getZs(float* hZs) { 
            int split_k_layers = S/R;
            int finalLayer = split_k_layers - 1;
            int finalOffset = finalLayer*A*m_maxBatch;
            getActivation(hZs, m_skipOutFinal + finalOffset, m_maxBatch*A); 
        } 
        void getZa(float* hZa) {
            int split_k_layers = A/R;
            int finalLayer = split_k_layers - 1;
            int finalOffset = finalLayer*A*m_maxBatch;
            getActivation(hZa, m_out + finalOffset, m_maxBatch*A); 
        }
        void getP(float* hP) { getActivation(hP, m_p, m_maxBatch*A); }
        void getYOut(int* yOut, int offset, int size, cudaStream_t stream = 0) {
            size_t cpy_pitch = m_maxSamples * sizeof(int); // spacing between chunk first elements
            size_t cpy_width = size * sizeof(int); // size of individual chunk
            size_t cpy_height = m_maxBatch;
            gpuErrChk(cudaMemcpy2DAsync(yOut + offset, cpy_pitch, m_yOut + offset, cpy_pitch, cpy_width, cpy_height, cudaMemcpyDeviceToHost, stream));
        }
        template<class Callback>
        bool run_chunks(int num_samples_per_chunk, Callback consume, int num_samples, int batch_size, int* yOut=NULL, int batch_size_per_block=1, bool dumpActivations=false, cudaStream_t stream = 0) {
            bool result = true;
            cudaStream_t stream_compute, stream_copy;
            if(!stream)
              cudaStreamCreate(&stream_compute);
            else
              stream_compute = stream;
            cudaStreamCreate(&stream_copy);
            m_num_samples_per_chunk = num_samples_per_chunk;
            int num_chunks = (num_samples + m_num_samples_per_chunk - 1) / m_num_samples_per_chunk;

            std::vector<cudaEvent_t> event_compute(num_chunks);
            std::vector<cudaEvent_t> event_copy(num_chunks);
            for (int j = 0; j < num_chunks; j++) {
              cudaEventCreateWithFlags(&(event_compute[j]), cudaEventDisableTiming);
              cudaEventCreateWithFlags(&(event_copy[j]), cudaEventDisableTiming);
            }

            for (int j = 0; j < num_chunks; j++) {

              int initSample = j*m_num_samples_per_chunk;
              if (j == num_chunks - 1) {
                m_num_samples_per_chunk = num_samples - initSample;
              }

              result = result && run_partial(initSample, num_samples, batch_size, NULL, batch_size_per_block, true, stream_compute);
              cudaEventRecord(event_compute[j], stream_compute);
              cudaStreamWaitEvent(stream_copy, event_compute[j], 0);
              if(yOut != NULL)
                getYOut(yOut, initSample, m_num_samples_per_chunk, stream_copy);
              cudaEventRecord(event_copy[j], stream_copy);
            }
            m_num_samples_per_chunk = num_samples_per_chunk;
            for (int j = 0; j < num_chunks; j++) {

              int initSample = j*m_num_samples_per_chunk;
              if (j == num_chunks - 1) {
                m_num_samples_per_chunk = num_samples - initSample;
              }
              cudaEventSynchronize(event_copy[j]);
              consume(yOut, initSample, m_num_samples_per_chunk);
            }
            m_num_samples_per_chunk = 0;
            for (int j = 0; j < num_chunks; j++) {
              cudaEventDestroy(event_compute[j]);
              cudaEventDestroy(event_copy[j]);
            }
            if(stream != stream_compute)
              cudaStreamDestroy(stream_compute);
            cudaStreamDestroy(stream_copy);
            return result;
        }

        bool run_partial(int init_sample, int num_samples, int batch_size, int* yOut=NULL, int batch_size_per_block=1, bool dumpActivations=false, cudaStream_t stream = 0) {

            Implementation impl = m_implementation;
            if (impl == AUTO) {
                if ((S == 2*R) && m_numLayers <= 20) {
                    impl = SINGLE_BLOCK;
                }
                else {
                    impl = DUAL_BLOCK;
                }
            }
            else if (impl == SINGLE_BLOCK) {
                assert(S<=4*R);
            }

            nv_wavenet_params<T_weight, T_data> params;
            params.num_samples = num_samples;
            params.init_sample = init_sample;
            params.num_samples_per_chunk = m_num_samples_per_chunk ? m_num_samples_per_chunk : num_samples;
            params.batch_size = batch_size;
            params.num_layers = m_numLayers;
            params.yInPrev = m_yInPrev;
            params.yInCur = m_yInCur;
            params.embedPrev = m_embedPrev;
            params.embedCur = m_embedCur;
            params.tanhEmbed = m_tanhEmbed;
            params.Wprev = m_Wprev;
            params.L = m_Lh;
            params.Wcur = m_Wcur;
            params.B = m_Bh;
            params.Wres = m_Wres;
            params.Bres = m_Bres;
            params.Wskip = m_Wskip;
            params.Bskip = m_Bskip;
            params.xt = m_XtIn;
            params.xtOut = m_XtOut;
            params.a_prev = m_aPrev;
            params.skip_in = m_skipIn;
            params.skip_out = m_skipOut;
            params.WskipOut = m_WskipOut;
            params.BskipOut = m_BskipOut;
            params.skipOutFinal = m_skipOutFinal;
            params.skipOutAccumulate = m_skipOutFinalAccumulate;
            params.Wout = m_Wout;
            params.Bout = m_Bout;
            params.out = m_out;
            params.outAccumulate = m_outAccumulate;
            params.p = m_p;
            params.outputSelectors = m_outputSelectors;
            params.yOut = m_yOut;
            params.dumpActivations = dumpActivations;
            params.maxDilation = m_maxDilation;

            params.h = m_h;
            params.hSample = m_hSample;
            params.ySample = m_ySample;

            bool result = false;

            if (impl == PERSISTENT) {
                assert(batch_size_per_block < 5);
                if (batch_size_per_block == 4) {
                    assert(batch_size%4==0);
                    result = launch_persistent<T_weight, T_data, R, S, A, 4>()(params, stream);
                }
                else if (batch_size_per_block == 3) {
                    assert(batch_size%3==0);
                    result =  launch_persistent<T_weight, T_data, R, S, A, 3>()(params, stream);
                }
                else if (batch_size_per_block == 2) {
                    assert(batch_size%2==0);
                    result =  launch_persistent<T_weight, T_data, R, S, A, 2>()(params, stream);
                }
                else {
                    result =  launch_persistent<T_weight, T_data, R, S, A, 1>()(params, stream);
                }
            } 
            else if (impl == MANYBLOCK_NONPERSISTENT) {
                assert(batch_size_per_block < 5);
                if (batch_size_per_block == 4) {
                    assert(batch_size%4==0);
                    result = launch_manyblock<false,T_weight, T_data, R, S, A, 4>()(params, stream);
                }
                else if (batch_size_per_block == 3) {
                    assert(batch_size%3==0);
                    result =  launch_manyblock<false,T_weight, T_data, R, S, A, 3>()(params, stream);
                }
                else if (batch_size_per_block == 2) {
                    assert(batch_size%2==0);
                    result =  launch_manyblock<false,T_weight, T_data, R, S, A, 2>()(params, stream);
                }
                else {
                    result =  launch_manyblock<false,T_weight, T_data, R, S, A, 1>()(params, stream);
                }
            }
            else if (R <= 64 && impl == DUAL_BLOCK) {
                assert(batch_size_per_block < 5);
                if (batch_size_per_block == 4) {
                    assert(batch_size%4==0);
                    result =  launch_dualBlock<T_weight, T_data, R, S, A, 4>()(params, stream);
                }
                else if (batch_size_per_block == 3) {
                    assert(batch_size%3==0);
                    result =  launch_dualBlock<T_weight, T_data, R, S, A, 3>()(params, stream);
                }
                else if (batch_size_per_block == 2) {
                    assert(batch_size%2==0);
                    result =  launch_dualBlock<T_weight, T_data, R, S, A, 2>()(params, stream);
                }
                else {
                    result =  launch_dualBlock<T_weight, T_data, R, S, A, 1>()(params, stream);
                }

            }
            else if (R <= 64){
                assert(batch_size_per_block < 5);
                if (batch_size_per_block == 4) {
                    assert(batch_size%4==0);
                    result =  launch_singleBlock<T_weight, T_data, R, S, A, 4>()(params, stream);
                }
                else if (batch_size_per_block == 3) {
                    assert(batch_size%3==0);
                    result =  launch_singleBlock<T_weight, T_data, R, S, A, 3>()(params, stream);
                }
                else if (batch_size_per_block == 2) {
                    assert(batch_size%2==0);
                    result =  launch_singleBlock<T_weight, T_data, R, S, A, 2>()(params, stream);
                }
                else {
                    result =  launch_singleBlock<T_weight, T_data, R, S, A, 1>()(params, stream);
                }
            }
            if (yOut != NULL) {
                gpuErrChk(cudaMemcpyAsync(yOut, m_yOut, m_maxSamples*m_maxBatch*sizeof(int), cudaMemcpyDeviceToHost, stream));
            }
            return result;
        }
        bool run(int num_samples, int batch_size, int* yOut=NULL, int batch_size_per_block=1, bool dumpActivations=false, cudaStream_t stream = 0) {
            m_num_samples_per_chunk = 0;
            return run_partial(0, num_samples, batch_size, yOut, batch_size_per_block, dumpActivations, stream);
        }
};

