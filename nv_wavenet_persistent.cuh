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

__device__ __forceinline__ bool isNegativeZero(float a) {
    int ret;
    asm volatile("{  set.eq.s32.b32 %0, %1, %2;}\n" : "=r"(ret) : "f"(a), "r"(0x80000000));
    return ret;
}

__device__ __forceinline__ bool isNegativeZero(half a){
    const __half_raw* a_raw_ptr = (reinterpret_cast<const __half_raw *>(&a) );
    int ret;
    asm volatile("{  set.eq.s32.b32 %0, %1, %2;}\n" : "=r"(ret) : "r"(0x0u + (*a_raw_ptr).x), "r"(0x00008000));
    return ret;
}

__device__ __forceinline__ float validate(float a) {
    return isNegativeZero(a) ? 0.f : a;
}

__device__ __forceinline__ half validate(half a) {
    return isNegativeZero(a) ? (half)0.f : a;
}

__device__ __forceinline__ void storeValidate(volatile half* y, int index, half val) {
    half* y_nv = (half*)y;
    y_nv[index] = validate(val);
}

__device__ __forceinline__ void storeValidate(volatile float* y, int index, float val) {
    y[index] = validate(val);
}

template <typename T_data, int R>
__global__ void initializeActivations(T_data* xt, T_data* h_out, T_data* a_prev, int num_layers, int batch_size) {
    assert(blockDim.x == R);

    int offset = blockIdx.x*blockDim.x + threadIdx.x;

    xt[offset] = -0.f;
    h_out[offset] = -0.f;

    a_prev[offset*2] = -0.f;
    a_prev[offset*2 + 1] = -0.f;
}

template <typename T_data>
__global__ void initializeActivationsGeneric(T_data* skipIn) {
    int offset = blockIdx.x*blockDim.x + threadIdx.x;
    skipIn[offset] = -0.f;
}

// Make sure all necessary clears are completed before processing a new sample.  Lock is per batch index.
template <int BATCH_UNROLL> 
__device__ __inline__ void sampleLockAcquire(int batch_offset, int sample, volatile int* sampleLock){
    if (threadIdx.x == 0) {
        bool valid = false;
        while (!valid) {
            valid = true;
#pragma unroll
            for (int u=0; u<BATCH_UNROLL; u++) {  
                valid &= (sampleLock[batch_offset+u]>=sample);
            }
        }
    }
    __syncthreads();
}

/* GEMM Tile -- M threads, with K weights held in registers */
template <typename T_weight, typename T_data, int M, int K, int N_UNROLL>
__device__ void nv_wavenet_persistent_GEMM_MxK(int thread_id, int num_samples, volatile int* ySample, int layer, int num_layers, int batch_size, T_weight* W, T_data* B, volatile T_data* act_in, T_data* act_out, volatile T_data* accum_in=NULL, int lda=M, int ldb=K, int ldc=M, bool doRelu=false) {
    int row = thread_id;
    const int WV = sizeof(T_weight)/sizeof(T_data);
    T_weight weights[K/WV];
    loadWeights<M,K>(weights,W,layer,row,lda);
    T_data accum[N_UNROLL];
    T_data bias = B ? B[layer*lda+row] : (T_data)0.f;

    __shared__ T_data act_in_sh[N_UNROLL][K];
    T_data act_in_reg[N_UNROLL];

    if (thread_id < M) {
        for (int sample=0; sample < num_samples; sample++) {
            for (int batch_offset = 0; batch_offset < batch_size; batch_offset += N_UNROLL) {
                // sampleLockacquire has a __syncthreads in it, so we don't need to worry about act_in_sh race
                sampleLockAcquire<N_UNROLL>(batch_offset, sample, ySample);
                if (row < K) {
                    bool valid = false;
                    while (!valid) {
                        valid = true;
#pragma unroll
                        for (int b=0; b<N_UNROLL; b++) {
                            act_in_reg[b] = loadVolatile(act_in,(batch_offset+b)*ldb + row);
                        }
#pragma unroll
                        for (int b=0; b<N_UNROLL; b++) {
                            valid &= !isNegativeZero(act_in_reg[b]);
                        }
                    }
#pragma unroll
                    for (int b=0; b<N_UNROLL; b++) {
                        act_in_sh[b][row] = act_in_reg[b];
                    }
                }
                __syncthreads();
                GEMM<K,2,N_UNROLL>(weights,act_in_sh,accum);
                if (accum_in) {
                    if (layer > 0) {
                        bool valid = false;
                        T_data accum_in_reg[N_UNROLL];
                        while (!valid) {
                            valid = true;
#pragma unroll
                            for (int b=0; b<N_UNROLL; b++) {
                                accum_in_reg[b] = loadVolatile(accum_in,(layer-1)*batch_size*ldc + (batch_offset+b)*ldc + row);
                            }
#pragma unroll
                            for (int b=0; b<N_UNROLL; b++) {
                                valid &= !isNegativeZero(accum_in_reg[b]);
                            }
                        }
#pragma unroll
                        for (int b=0; b<N_UNROLL; b++) {
                            accum[b] += accum_in_reg[b];
                        }
                    }
                }
#pragma unroll
                for (int b=0; b<N_UNROLL; b++) {
                    accum[b] += bias;
                    if (doRelu) accum[b] = relu(accum[b]);
                    act_out[layer*batch_size*ldc + (batch_offset+b)*ldc + row] = accum[b];
                    if (accum_in)  {
                        storeValidate(accum_in,layer*batch_size*ldc + (batch_offset+b)*ldc + row,accum[b]);
                    }
                }
            }
        }
    }
}

/* Tiled GEMM, where each tile has TILE_M threads and TILE_K weights */
template <typename T_weight, typename T_data, int TILE_M, int TILE_K, int BATCH_UNROLL>
__device__ void nv_wavenet_persistent_GEMM(int thread_id, int num_samples, volatile int* ySample, int tile_id, int batch_size, T_weight* W, T_data* B, volatile T_data* act_in, T_data* act_out, T_data* accum_in, int gemm_m, int gemm_k, bool doRelu=false) {

   int tiles_m = gemm_m / TILE_M;
   int tiles_k = gemm_k / TILE_K;

   int tile_id_m = tile_id % tiles_m;
   int tile_id_k = tile_id / tiles_m;

   int tile_offset_m = tile_id_m*TILE_M;
   int tile_offset_k = tile_id_k*TILE_K;

   T_data* bias = (tile_id_k == 0) ? B + tile_offset_m : NULL;

   nv_wavenet_persistent_GEMM_MxK<T_weight, T_data, TILE_M, TILE_K, BATCH_UNROLL>(thread_id, num_samples, ySample, tile_id_k, tiles_k, batch_size, W + tile_offset_m, bias, act_in + tile_offset_k, act_out + tile_offset_m, accum_in + tile_offset_m, gemm_m, gemm_k, gemm_m, doRelu && (tile_id_k == tiles_k-1)); 
}

template <typename T_weight, typename T_data, int R, int BATCH_UNROLL>
__device__ void nv_wavenet_persistent_prev(int row, int num_samples, volatile int* ySample, int layer, int num_layers, int batch_size, int maxDilation, T_weight* Wprev, T_data* a_prev, volatile T_data* xt) {
    const int WV = sizeof(T_weight)/sizeof(T_data);
    T_weight weights[R/WV];
    loadWeights<2*R,R>(weights,Wprev,layer,row);
    T_data accum[BATCH_UNROLL];
    __shared__ T_data xtmd_sh[BATCH_UNROLL][R];

    int dilation = 1;
    for (int l=1; l<=layer; l++) {
        dilation = dilation << 1;
        if (dilation > maxDilation) dilation = 1;
    }

    if (row < 2*R) {
        for (int sample=0; sample<num_samples; sample++) {
            int sample_offset = (sample - dilation) % (maxDilation+1);
            volatile T_data* xtmd = xt + sample_offset*(num_layers+1)*R*batch_size;
            for (int batch_offset = 0; batch_offset < batch_size; batch_offset += BATCH_UNROLL) {
                sampleLockAcquire<BATCH_UNROLL>(batch_offset,sample,ySample);
                if (row < R) {
#pragma unroll
                    for (int b=0; b<BATCH_UNROLL; b++) {
                        xtmd_sh[b][row] = (dilation <= sample) ? loadVolatile(xtmd,layer*batch_size*R + (batch_offset+b)*R + row) : (T_data)0.f;
                    }
                }
                __syncthreads();
                GEMM<R,2,BATCH_UNROLL>(weights, xtmd_sh, accum);
#pragma unroll
                for (int b=0; b<BATCH_UNROLL; b++) {
                    a_prev[layer*batch_size*2*R + (batch_offset+b)*2*R + threadIdx.x] = accum[b]; 
                }
            }
        }
    }
}

template <typename T_weight, typename T_data, int R, int BATCH_UNROLL>
__device__ void nv_wavenet_persistent_cur(int row, int num_samples, volatile int* ySample, int layer, int num_layers, int batch_size, int maxDilation, T_weight* Wcur, T_data* B, T_data* L, T_data a_cur_sh[BATCH_UNROLL][2*R], volatile T_data* a_prev, volatile T_data* xt, int* yInPrev, int* yInCur, T_data* embedPrev, T_data* embedCur, bool tanhEmbed) {
    const int WV = sizeof(T_weight)/sizeof(T_data);
    T_weight weights[R/WV];
    loadWeights<2*R,R>(weights,Wcur,layer,row);
    T_data accum[BATCH_UNROLL];
    T_data bias = B[layer*2*R+row];
    T_data a_prev_reg[BATCH_UNROLL];
    T_data xt_in[BATCH_UNROLL];
    for (int sample=0; sample<num_samples; sample++) {
        __syncthreads(); // Wait for initial sample lock
        volatile T_data* Xt = xt + (sample%(maxDilation+1))*(num_layers+1)*R*batch_size;
        for (int batch_offset = 0; batch_offset < batch_size; batch_offset += BATCH_UNROLL) {
            T_data conditioning[BATCH_UNROLL];
#pragma unroll
            for (int b=0; b<BATCH_UNROLL; b++) {
                conditioning[b] = L[sample*num_layers*batch_size*2*R + layer*batch_size*2*R + (batch_offset+b)*2*R + row];
            }
            __shared__ T_data xt_sh[BATCH_UNROLL][R];
            if (row < R) {
                if (layer == 0) {
                    // Embedding
                    int yPrev[BATCH_UNROLL];
                    int yCur[BATCH_UNROLL];
#pragma unroll
                    for (int b=0; b<BATCH_UNROLL; b++) {
                        yPrev[b] = yInPrev[batch_offset+b];
                        yCur[b] = yInCur[batch_offset+b];
                        T_data embedded = embedPrev[yPrev[b]*R + row] + embedCur[yCur[b]*R + row];
                        if (tanhEmbed) embedded = _tanh(embedded);
                        xt_sh[b][row] = embedded;
                        storeValidate(Xt, layer*batch_size*R + (batch_offset+b)*R + row, embedded);
                    }
                    // Make Xt visible before we write h, so that clears don't race ahead
                    // This is only needed for the embedding write, since it's read by the same block -- 
                    //  all other Xt writes get read by different blocks before they write h.  Since
                    //  the clears depend on h, then we know that the Xt writes are globally visible.
                    __threadfence();
                }
            }
            bool valid = false;
            int a_prev_offset = layer*batch_size*2*R + batch_offset*2*R + row;
            int xt_offset = layer*batch_size*R + batch_offset*R + row;
            // Do redundant loads in upper half to avoid branch in polling loop.
            if (row >= R) xt_offset -= R;
            while (!valid) {
                valid = true;
#pragma unroll
                for (int b=0; b<BATCH_UNROLL; b++) {
                    a_prev_reg[b] = loadVolatile(a_prev,a_prev_offset+b*2*R);
                    xt_in[b] = loadVolatile(Xt,xt_offset+b*R);
                }
#pragma unroll
                for (int b=0; b<BATCH_UNROLL; b++) {
                    valid &= !isNegativeZero(a_prev_reg[b]);
                    valid &= !isNegativeZero(xt_in[b]);
                }
            }
            if (row < R) {
#pragma unroll
                for (int b=0; b<BATCH_UNROLL; b++) {
                    xt_sh[b][row] = xt_in[b];
                }
            }
            namedBarrierSync(1,2*R);
            GEMM<R,2,BATCH_UNROLL>(weights,xt_sh,accum);
#pragma unroll
            for (int b=0; b<BATCH_UNROLL; b++) { 
                accum[b] += a_prev_reg[b];
                accum[b] += bias; 
                accum[b] += conditioning[b];
                T_data val = (row < R) ? _tanh(accum[b]) : sigmoid(accum[b]);
                a_cur_sh[b][row] = val;
            }
            namedBarrierSync(3,3*R); // a_cur_sh produced
            __syncthreads(); // a_cur_sh consumed
        }
    }
}

template <typename T_weight, typename T_data, int R, int BATCH_UNROLL>
__device__ void nv_wavenet_persistent_res(int row, int num_samples, volatile int* ySample, int layer, int num_layers, int batch_size, int maxDilation, T_weight* Wres, T_data* Bres, T_data a_cur_sh[BATCH_UNROLL][2*R], T_data* xt, T_data* h, T_data* xtOut, bool dumpActivations) {
    const int WV = sizeof(T_weight)/sizeof(T_data);
    T_weight weights[R/WV];
    T_data bias = Bres[layer*R+row];
    T_data accum[BATCH_UNROLL];
    __shared__ T_data h_sh[BATCH_UNROLL][R];
    loadWeights<R,R>(weights,Wres,layer,row);
    for (int sample=0; sample<num_samples; sample++) {
        __syncthreads(); // Wait for initial sample lock
        for (int batch_offset = 0; batch_offset < batch_size; batch_offset += BATCH_UNROLL) {
            namedBarrierSync(3,3*R); // a_cur_sh produced, h_sh consumed
#pragma unroll
            for (int b=0; b<BATCH_UNROLL; b++) {
                T_data val = a_cur_sh[b][row] * a_cur_sh[b][row + R];
                h_sh[b][row] = val;
                h[layer*batch_size*R + (batch_offset+b)*R + row] = validate(val);
            }
            __syncthreads(); // a_cur_sh consumed, h_sh produced
            GEMM<R,2,BATCH_UNROLL>(weights,h_sh,accum);
            T_data* Xt = xt + (sample%(maxDilation+1))*(num_layers+1)*R*batch_size;
#pragma unroll
            for (int b=0; b<BATCH_UNROLL; b++) { 
                accum[b] += bias; 
                accum[b] += Xt[layer*batch_size*R + (batch_offset+b)*R + row];
                Xt[(layer+1)*batch_size*R + (batch_offset+b)*R + row] = accum[b];
                if (dumpActivations) xtOut[layer*batch_size*R + (batch_offset+b)*R + row] = accum[b];
            }
        }
    }
}

template <typename T_weight, typename T_data, int R, int BATCH_UNROLL>
__device__ void nv_wavenet_persistent_cur_res(int thread_id, int num_samples, volatile int* ySample, int layer, int num_layers, int batch_size, int maxDilation, T_weight* Wcur, T_data* B, T_data* L, T_weight* Wres, T_data* Bres, T_data* a_prev, T_data* xt, T_data* h, T_data* xtOut, bool dumpActivations, int* yInPrev, int* yInCur, T_data* embedPrev, T_data* embedCur, bool tanhEmbed) {
    __shared__ T_data a_cur_sh[BATCH_UNROLL][2*R];
    if (thread_id < R) {
        for (int sample=0; sample<num_samples; sample++) {
            sampleLockAcquire<BATCH_UNROLL>(0,sample,ySample);
            for (int batch_offset = 0; batch_offset < batch_size; batch_offset += BATCH_UNROLL) {
                if (batch_offset+BATCH_UNROLL<batch_size) sampleLockAcquire<BATCH_UNROLL>(batch_offset+BATCH_UNROLL,sample,ySample);
                else __syncthreads();
            }
        }
    }
    else if (thread_id < 3*R) {
        int row = thread_id - R;
        nv_wavenet_persistent_cur<T_weight, T_data, R, BATCH_UNROLL>(row, num_samples, ySample, layer, num_layers, batch_size, maxDilation, Wcur, B, L, a_cur_sh, a_prev, xt, yInPrev, yInCur, embedPrev, embedCur, tanhEmbed); 
    }
    else if (thread_id < 4*R) {
        int row = thread_id - 3*R;
        nv_wavenet_persistent_res<T_weight, T_data, R, BATCH_UNROLL>(row, num_samples, ySample, layer, num_layers, batch_size, maxDilation, Wres, Bres, a_cur_sh, xt, h, xtOut, dumpActivations);
    }
}

template <typename T_weight, typename T_data, int R, int S, int A, int BATCH_UNROLL>
__device__ void nv_wavenet_persistent_softmax(int block_id, int batch_size, int num_layers, int num_samples, int maxDilation, volatile T_data* outAccumulate, float* outputSelectors, T_data* p, int* yOut, int* yInPrev, int* yInCur, volatile int* ySample, T_data* xt, T_data* a_prev, T_data* h, T_data* skip_out, T_data* skipOutAccumulate, bool dumpActivations) {
    for (int sample = 0; sample < num_samples; sample++) {
        __shared__ T_data out_sh[BATCH_UNROLL][A];
        __shared__ T_data p_sh[BATCH_UNROLL][A];
        __shared__ int yOut_sh[BATCH_UNROLL];

        int col = block_id*BATCH_UNROLL;

        const int NUM_THREADS=2*R;
        if (threadIdx.x < NUM_THREADS) {

            const int ROWS_PER_THREAD = A/NUM_THREADS;
            T_data out_reg[BATCH_UNROLL][ROWS_PER_THREAD];
            bool valid = false;
            while (!valid) {
                valid = true;
#pragma unroll
                for (int u=0; u<BATCH_UNROLL; u++) {
                    for (int r=0; r<ROWS_PER_THREAD; r++) {
                        int row = threadIdx.x*ROWS_PER_THREAD + r;
                        out_reg[u][r] = loadVolatile(outAccumulate,(A/R-1)*batch_size*A + (col+u)*A + row);
                    }
                }
#pragma unroll
                for (int u=0; u<BATCH_UNROLL; u++) {
                    for (int r=0; r<ROWS_PER_THREAD; r++) {
                        valid &= !isNegativeZero(out_reg[u][r]);
                    }
                }
            }
#pragma unroll
            for (int u=0; u<BATCH_UNROLL; u++) {
                for (int r=0; r<ROWS_PER_THREAD; r++) {
                    out_sh[u][threadIdx.x*ROWS_PER_THREAD+r] = out_reg[u][r];
                }
            }
        }

        __syncthreads();

        if (threadIdx.x < NUM_THREADS) {
            softmax_select<T_data, NUM_THREADS, A,BATCH_UNROLL>(0,BATCH_UNROLL, (T_data*)out_sh, dumpActivations ? (T_data*)p_sh : NULL, outputSelectors + sample*batch_size + col, yOut_sh, 1, NUM_THREADS);

            namedBarrierSync(1,NUM_THREADS);

#pragma unroll
            for (int u=0; u<BATCH_UNROLL; u++) {
                if (dumpActivations) {
                    for (int i=threadIdx.x; i<A; i += 2*R){
                        p[(col+u)*A + i] = p_sh[u][i];
                    }
                }

                if (threadIdx.x == 0) {
                    yOut[(col+u)*num_samples + sample] = yOut_sh[u];
                    yInPrev[col+u] = yInCur[col+u];
                    yInCur[col+u] = yOut_sh[u];
                }
            }
        }
        else if (threadIdx.x < NUM_THREADS+R && sample+1<num_samples) {
            int thread_id = threadIdx.x - NUM_THREADS;
            volatile T_data* Xt = xt + ((sample+1)%(maxDilation+1))*(num_layers+1)*R*batch_size;
            for (int l=0; l<num_layers; l++) {
                for (int u=0; u<BATCH_UNROLL; u++) {
                    storeVolatile(Xt,l*batch_size*R + (col+u)*R + thread_id,-0.f);
                    storeVolatile(h,l*batch_size*R + (col+u)*R + thread_id,-0.f);
                    a_prev[l*batch_size*2*R + (col+u)*2*R + thread_id] = -0.f;
                    a_prev[l*batch_size*2*R + (col+u)*2*R + thread_id + R] = -0.f;
                    for (int i=0;i<S/R;i++) {
                        skip_out[l*batch_size*S + (col+u)*S + i*R + thread_id] = -0.f;
                    }
                }
            }
            for (int l=0; l<S/R; l++) {
                for (int i=0; i<A/R; i++) {
                    for (int u=0; u<BATCH_UNROLL; u++) {
                        skipOutAccumulate[l*batch_size*A + (col+u)*A + i*R + thread_id] = -0.f;        
                    }
                }
            }
            for (int l=0; l<A/R; l++) {
                for (int i=0; i<A/R; i++) {
                    for (int u=0; u<BATCH_UNROLL; u++) {
                        storeVolatile(outAccumulate,l*batch_size*A + (col+u)*A + i*R + thread_id,-0.f);        
                    }
                }
            }
        }

        // Make sure all the clears are visible before we advance the sample lock
        __threadfence();
        __syncthreads();
        if (threadIdx.x == 0) {
#pragma unroll
            for (int u=0; u<BATCH_UNROLL; u++) {
                ySample[col+u] = sample+1;
            }
        }
    }
}

template <typename T_weight, typename T_data, int R, int S, int A, int BATCH_UNROLL>
__global__ void nv_wavenet_persistent(nv_wavenet_params<T_weight, T_data> params) {
    int prev_blocks = params.num_layers;
    int cur_blocks = params.num_layers;
    const int S_TILE = S < 4*R ? S : 4*R;
    int s_tiles = S / S_TILE;
    int skip_blocks = params.num_layers * s_tiles;
    int Zs_blocks = (A/(4*R)) * (S/R);
    int Za_blocks = (A/(4*R)) * (A/R);
    //int softmax_blocks = params.batch_size;
    int thread_id = threadIdx.x;

    if (blockIdx.x < prev_blocks) {
        // Prev
        int layer = blockIdx.x;
        nv_wavenet_persistent_prev<T_weight, T_data, R, BATCH_UNROLL>(thread_id, params.num_samples, params.ySample, layer, params.num_layers, params.batch_size, params.maxDilation, params.Wprev, params.a_prev, params.xt);
    }
    else if (blockIdx.x < prev_blocks + cur_blocks) {
        // Cur
        int layer = blockIdx.x - prev_blocks;
        nv_wavenet_persistent_cur_res<T_weight, T_data, R, BATCH_UNROLL>(thread_id, params.num_samples, params.ySample, layer, params.num_layers, params.batch_size, params.maxDilation, params.Wcur, params.B, params.L, params.Wres, params.Bres, params.a_prev, params.xt, params.h, params.xtOut, params.dumpActivations, params.yInPrev, params.yInCur, params.embedPrev, params.embedCur, params.tanhEmbed);
    }
    else if (blockIdx.x < prev_blocks + cur_blocks + skip_blocks) {
        // Skip
        int block_id = blockIdx.x - prev_blocks - cur_blocks;
        int layer = block_id*s_tiles;
        int tile = block_id%s_tiles;
        int tile_offset = tile*S_TILE;
        nv_wavenet_persistent_GEMM_MxK<T_weight, T_data, S_TILE, R, BATCH_UNROLL>(thread_id, params.num_samples, params.ySample,layer, params.num_layers, params.batch_size, params.Wskip + tile_offset, params.Bskip + tile_offset, params.h + layer*params.batch_size*R, params.skip_out + tile_offset, params.skip_out + tile_offset, S, R, S, layer==params.num_layers-1);
    }
    // AxS
    else if (blockIdx.x < prev_blocks + cur_blocks + skip_blocks + Zs_blocks) {
        int tile_id = blockIdx.x - prev_blocks - cur_blocks - skip_blocks;
        nv_wavenet_persistent_GEMM<T_weight, T_data, 4*R, R, BATCH_UNROLL>(thread_id, params.num_samples, params.ySample, tile_id, params.batch_size, params.WskipOut, params.BskipOut, params.skip_out + (params.num_layers-1)*params.batch_size*S, params.skipOutFinal, params.skipOutAccumulate, A, S, true);
    }
    else if (blockIdx.x < prev_blocks + cur_blocks + skip_blocks + Zs_blocks + Za_blocks) {
        int tile_id = blockIdx.x - prev_blocks - cur_blocks - skip_blocks - Zs_blocks;
        nv_wavenet_persistent_GEMM<T_weight, T_data, 4*R, R, BATCH_UNROLL>(thread_id, params.num_samples, params.ySample, tile_id, params.batch_size, params.Wout, params.Bout, params.skipOutAccumulate + (S/R-1)*A*params.batch_size, params.out, params.outAccumulate, A, A);
    }
    else {
        int block_id = blockIdx.x - prev_blocks - cur_blocks - skip_blocks - Zs_blocks - Za_blocks;
        nv_wavenet_persistent_softmax<T_weight, T_data, R, S, A, 1>(block_id, params.batch_size, params.num_layers, params.num_samples, params.maxDilation, params.outAccumulate, params.outputSelectors, params.p, params.yOut, params.yInPrev, params.yInCur, params.ySample, params.xt, params.a_prev, params.h, params.skip_out, params.skipOutAccumulate, params.dumpActivations);
    }
}

template <typename T_weight, typename T_data, int R, int S, int A, int BATCH_UNROLL>
bool launch_persistent(nv_wavenet_params<T_weight, T_data> params, cudaStream_t stream) {
    int prev_blocks = params.num_layers;
    int cur_blocks = params.num_layers;
    if (S<4*R) assert (S%R==0); else assert(S%4*R==0);
    assert(A>=4*R);
    const int S_TILE = S < 4*R ? S : 4*R;
    int s_tiles = S / S_TILE;
    int skip_blocks = params.num_layers * s_tiles;
    int Zs_blocks = (A/(4*R)) * (S/R);
    int Za_blocks = (A/(4*R)) * (A/R);
    int softmax_blocks = params.batch_size;
    dim3 grid(prev_blocks + cur_blocks + skip_blocks + Zs_blocks + Za_blocks + softmax_blocks);
    dim3 block(4*R);
    if (S > 4*R) block.x = S;
    int occ = getOccupancy(0, block.x*block.y*block.z,(void*)nv_wavenet_persistent<T_weight, T_data, R, S, A, BATCH_UNROLL>);
    printf("%d blocks, %d blocks per SM\n", grid.x, occ);
    assert(occ>0);
    gpuErrChk(cudaMemset((void*)params.hSample,0,params.num_layers*params.batch_size*sizeof(int)));
    gpuErrChk(cudaMemset((void*)params.ySample,0,params.batch_size*sizeof(int)));
    initializeActivations<T_data,R><<<params.num_layers*params.batch_size,R,0,stream>>>(params.xt, params.h, params.a_prev, params.num_layers, params.batch_size);
    initializeActivationsGeneric<T_data><<<(params.maxDilation+1)*(params.num_layers+1)*params.batch_size,R,0,stream>>>(params.xt);
    initializeActivationsGeneric<T_data><<<params.num_layers*params.batch_size,S,0,stream>>>(params.skip_out);
    initializeActivationsGeneric<T_data><<<(S/R)*params.batch_size,A,0,stream>>>(params.skipOutAccumulate);
    initializeActivationsGeneric<T_data><<<(A/R)*params.batch_size,A,0,stream>>>(params.outAccumulate);
    void* p_params = {&params};
    cudaError_t code = cudaLaunchCooperativeKernel((void*)nv_wavenet_persistent<T_weight,T_data,R,S,A,BATCH_UNROLL>, grid, block, &p_params, 0, stream);
    gpuAssert(code, __FILE__, __LINE__, false);
    return code == cudaSuccess;
}
