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

#include "matrix_math.cuh"
#include "softmax.cuh"

__device__ __inline__ unsigned long long int gclock64() {
    unsigned long long int rv;
    asm volatile ( "mov.u64 %0, %%globaltimer;" : "=l"(rv) );
    return rv;
}

template <typename T_weight, typename T_data, int R, int S, int BATCH_UNROLL>
__device__ void nv_wavenet_dualBlock_skip(int sample, int num_layers, int batch_offset, int batch_size, T_weight* Wskip, T_data* Bskip, T_data skip_out_sh[BATCH_UNROLL][S], T_data* skip_out, bool dumpActivations, volatile T_data* h, volatile int* hSample) {
    const int WV = sizeof(T_weight)/sizeof(T_data);
    T_weight weights[R/WV];
    T_data accum[BATCH_UNROLL];
    T_data skip_accum_last[BATCH_UNROLL];

    __shared__ T_data h_sh[2][BATCH_UNROLL][R];

    for (int b=0; b<BATCH_UNROLL; b++) {
        skip_accum_last[b] = 0.f;
    }

    int ping_pong = 0;

    if (threadIdx.x < 32) {
        for (int layer=0; layer<num_layers; layer++) {
            int row = threadIdx.x;
            if (row < BATCH_UNROLL) while (hSample[layer*batch_size + batch_offset + row] <= sample);
            __syncthreads();
        }
        __syncthreads();
    }
    else if (threadIdx.x >= 32 && threadIdx.x < 32 + R) {
        __syncthreads();
        for (int layer=0; layer<num_layers; layer++) {
            int row = threadIdx.x - 32;
    #pragma unroll
            for (int b=0; b<BATCH_UNROLL; b++) {
                h_sh[ping_pong][b][row] = loadVolatile(h,layer*batch_size*R + (batch_offset+b)*R + row);
            }
            __syncthreads();
            ping_pong = ping_pong ^ 1;
        }
    }
    else if (threadIdx.x >= 32+R && threadIdx.x < S + 32 + R) {
        __syncthreads();
        for (int layer=0; layer<num_layers; layer++) {
            __syncthreads();
            int row = threadIdx.x - (32+R);
            T_data bias = Bskip[layer*S + row];
            loadWeights<S,R>(weights,Wskip,layer,row);
            GEMM<R,2,BATCH_UNROLL>(weights,h_sh[ping_pong],accum);
            for (int b=0; b<BATCH_UNROLL; b++) { 
                accum[b] += bias;
                T_data val = accum[b] + skip_accum_last[b];
                skip_accum_last[b] += accum[b];
                skip_out_sh[b][row] = val;
                if (dumpActivations) skip_out[layer*batch_size*S + (batch_offset+b)*S + row] = val;
            }
            ping_pong = ping_pong ^ 1;
        }
    }
    else {
        for (int layer=0; layer<num_layers; layer++) {
            __syncthreads();
        } 
        __syncthreads();
    }

}

template <typename T_weight, typename T_data, int R, int S, int A, int BATCH_UNROLL>
__device__ void nv_wavenet_dualBlock_A(nv_wavenet_params<T_weight, T_data> params, int batch_offset) {

    __shared__ T_data xt_sh[BATCH_UNROLL][R];
    __shared__ T_data a_cur_sh[BATCH_UNROLL][2*R];
    __shared__ T_data h_sh[BATCH_UNROLL][R];

    for (int sample = 0; sample < params.num_samples; sample++) {

        // Pipeline the prev computation with final layers of prior sample
        nv_wavenet_prev<T_weight, T_data, R, BATCH_UNROLL>(sample, threadIdx.x, params.num_layers, params.maxDilation, batch_offset, params.batch_size, params.Wprev, params.L, params.xt, params.a_prev, params.dumpActivations);

        uint64_t prev = (threadIdx.x == 0) ? gclock64() : 0;

        // Now wait for the prior sample to be computed
        if (threadIdx.x < BATCH_UNROLL) {
            while (params.ySample[batch_offset + threadIdx.x] < sample);
        }

        __syncthreads();

        // Begin current sample

        // Embedding
        if (threadIdx.x < R) {
            int row = threadIdx.x;
            int yPrev[BATCH_UNROLL];
            int yCur[BATCH_UNROLL];
            for (int b=0; b<BATCH_UNROLL; b++) {
                yPrev[b] = params.yInPrev[batch_offset+b];
                yCur[b] = params.yInCur[batch_offset+b];

                T_data val = params.embedPrev[yPrev[b]*R + row] + params.embedCur[yCur[b]*R + row];
                if (params.tanhEmbed) val = _tanh(val);
                xt_sh[b][row] = val;
                T_data* Xt = params.xt + (sample%(params.maxDilation+1))*(params.num_layers+1)*R*params.batch_size;
                Xt[(batch_offset+b)*R + row] = val;
            }
        }

        __syncthreads();

        if (threadIdx.x < 2*R) {
            int row = threadIdx.x;
            nv_wavenet_cur<T_weight, T_data, R, BATCH_UNROLL>(sample, row, params.num_layers, batch_offset, params.batch_size, params.Wcur, params.B, params.L, xt_sh, a_cur_sh, params.a_prev);
        }
        else if (threadIdx.x < 3*R) {
            int row = threadIdx.x - 2*R;
            nv_wavenet_pointwise<T_weight, T_data, R, S, BATCH_UNROLL, true>(sample, row, params.num_layers, batch_offset, params.batch_size, params.xtmd, xt_sh, a_cur_sh, h_sh, params.h, params.hSample);
        }
        else if (threadIdx.x < 4*R) {
            int row = threadIdx.x - 3*R;
            nv_wavenet_res<T_weight, T_data, R, S, BATCH_UNROLL, true>(sample, row, params.num_layers, params.maxDilation, batch_offset, params.batch_size, params.Wres, params.Bres, h_sh, xt_sh, params.xt, params.xtOut, params.dumpActivations);
        }
        else {
            for (int layer=0; layer<params.num_layers;layer++) {
                __syncthreads();
            }
        }

    }

}

template <typename T_weight, typename T_data, int R, int S, int A, int BATCH_UNROLL>
__device__ void nv_wavenet_dualBlock_B(nv_wavenet_params<T_weight, T_data> params, int batch_offset) {
    for (int sample = 0; sample < params.num_samples; sample++) {

        int row = threadIdx.x;

        __shared__ T_data skip_out_sh[BATCH_UNROLL][S];

        nv_wavenet_dualBlock_skip<T_weight, T_data, R, S, BATCH_UNROLL>(sample, params.num_layers, batch_offset, params.batch_size, params.Wskip, params.Bskip, skip_out_sh, params.skip_out, params.dumpActivations, params.h, params.hSample);

        __syncthreads();

        const int WV = sizeof(T_weight)/sizeof(T_data);
        T_weight weights[R/WV];
        T_data accum[BATCH_UNROLL];

        __shared__ T_data out_sh[BATCH_UNROLL][A];
        __shared__ T_data skip_out_final_sh[BATCH_UNROLL][A];

        const int M = 4*R;

        T_data zero = 0.f;

        // relu
        for (int r = threadIdx.x; r < S; r += blockDim.x) {
            for (int b=0; b<BATCH_UNROLL; b++) {
                T_data d = skip_out_sh[b][r];
                skip_out_sh[b][r] = d < zero ? zero : d;
            }
        }
        __syncthreads();

        if (threadIdx.x < M) {
            // SkipOut: AxS
            for (int tile_m = 0; tile_m < A/M; tile_m++) {
                T_data bias = params.BskipOut[tile_m*M+row];
                T_data split_accum[BATCH_UNROLL];
                for (int b=0; b<BATCH_UNROLL; b++) {
                    split_accum[b] = 0.f; 
                }
                for (int tile_k = 0; tile_k < S/R; tile_k++) {
                    loadWeights<M,R>(weights, params.WskipOut + tile_m*M,  tile_k, threadIdx.x, A);
                    T_data activations[BATCH_UNROLL][R];
                    for (int b=0; b<BATCH_UNROLL; b++) {
                        for (int i=0; i<R; i++) {
                            activations[b][i] = skip_out_sh[b][tile_k*R + i];
                        }
                    }
                    GEMM<R,2,BATCH_UNROLL>(weights,activations,accum);
                    for (int b=0; b<BATCH_UNROLL; b++) {
                        split_accum[b] += accum[b];
                    }
                }
                for (int b=0; b<BATCH_UNROLL; b++) {
                    int finalLayer = S/R - 1;
                    split_accum[b] += bias;
                    skip_out_final_sh[b][tile_m*M + row] = split_accum[b] < zero ? zero : split_accum[b]; // relu
                    if (params.dumpActivations) params.skipOutFinal[finalLayer*params.batch_size*A + (batch_offset+b)*A + tile_m*M + row] = split_accum[b];
                }
            }
        }


        __syncthreads();

        if (threadIdx.x < M) {
            // Out: AxA
            for (int tile_m = 0; tile_m < A/M; tile_m++) {
                T_data bias = params.Bout[tile_m*M+row];
                T_data split_accum[BATCH_UNROLL];
                for (int b=0; b<BATCH_UNROLL; b++) {
                    split_accum[b] = 0.f; 
                }
                for (int tile_k = 0; tile_k < A/R; tile_k++) {
                    loadWeights<M,R>(weights, params.Wout + tile_m*M, tile_k, threadIdx.x, A);
                    T_data activations[BATCH_UNROLL][R];
                    for (int b=0; b<BATCH_UNROLL; b++) {
                        for (int i=0; i<R; i++) {
                            activations[b][i] = skip_out_final_sh[b][tile_k*R + i];
                        }
                    }
                    GEMM<R,2,BATCH_UNROLL>(weights,activations,accum);
                    for (int b=0; b<BATCH_UNROLL; b++) {
                        split_accum[b] += accum[b];
                    }
                }
                for (int b=0; b<BATCH_UNROLL; b++) {
                    int finalLayer = A/R - 1;
                    split_accum[b] += bias;
                    out_sh[b][tile_m*M + row] = split_accum[b];
                    if (params.dumpActivations) params.out[finalLayer*params.batch_size*A + (batch_offset+b)*A + tile_m*M + row] = split_accum[b];
                }
            }
        }

        __syncthreads();

        //__shared__ T_data p_sh[BATCH_UNROLL][A];
        T_data (*p_sh)[A] = skip_out_final_sh;

        __shared__ int yOut_sh[BATCH_UNROLL];

        if (threadIdx.x < M) {
            softmax_select<T_data, 4*R, A,BATCH_UNROLL>(0,BATCH_UNROLL, (T_data*)out_sh, params.dumpActivations ? (T_data*)p_sh : NULL, params.outputSelectors + sample*params.batch_size + batch_offset, yOut_sh, 1, 4*R);
        }

        __syncthreads();

        if (threadIdx.x < 4*R) {
            for (int u=0; u<BATCH_UNROLL; u++) {
                if (params.dumpActivations) {
                    for (int i=threadIdx.x; i<A; i += M) {
                        params.p[(batch_offset+u)*A + i] = p_sh[u][i];
                    }
                }
            }
        }

        // Now that we're done, prepare for next sample: yInPrev = yInCur, yIn = yOut
        if (threadIdx.x < BATCH_UNROLL) {
            int u = threadIdx.x;
            params.yOut[(batch_offset+u)*params.num_samples + sample] = yOut_sh[u];
            params.yInPrev[batch_offset+u] = params.yInCur[batch_offset+u];
            params.yInCur[batch_offset+u] = yOut_sh[u];
        }

        __syncthreads();
        __threadfence();
        if (row < BATCH_UNROLL) {
            params.ySample[batch_offset+row] = sample+1;
        }

    }
}

template <typename T_weight, typename T_data, int R, int S, int A, int BATCH_UNROLL>
__global__ void nv_wavenet_dualBlock(nv_wavenet_params<T_weight, T_data> params) {

    int batch_offset = blockIdx.x/2 * BATCH_UNROLL;
    int is_skip = blockIdx.x & 1;

    if (is_skip) nv_wavenet_dualBlock_B<T_weight, T_data, R, S, A, BATCH_UNROLL>(params, batch_offset);
    else         nv_wavenet_dualBlock_A<T_weight, T_data, R, S, A, BATCH_UNROLL>(params, batch_offset);

}

template <typename T_weight, typename T_data, int R, int S, int A, int BATCH_UNROLL>
bool launch_dualBlock(nv_wavenet_params<T_weight, T_data> params, cudaStream_t stream) {
    assert(BATCH_UNROLL <= 32); //Single-thread-per-batch things get parallelized across a warp
    dim3 grid(2*params.batch_size/BATCH_UNROLL);
    dim3 block(S + R + 32);
    if (4*R > block.x) block.x = 4*R;
    int occ = getOccupancy(0, block.x*block.y*block.z,(void*)nv_wavenet_dualBlock<T_weight, T_data, R, S, A, BATCH_UNROLL>);
    assert(occ>0);
    gpuErrChk(cudaMemset((void*)params.hSample,0,params.num_layers*params.batch_size*sizeof(int)));
    gpuErrChk(cudaMemset((void*)params.ySample,0,params.batch_size*sizeof(int)));
    // Since the two CTAs are communicating, launch as a cooperative kernel
    void* p_params = {&params};
    cudaError_t code = cudaLaunchCooperativeKernel((void*)nv_wavenet_dualBlock<T_weight,T_data,R,S,A,BATCH_UNROLL>, grid, block, &p_params, 0, stream);
    gpuAssert(code, __FILE__, __LINE__, false);
    return code == cudaSuccess;
}
