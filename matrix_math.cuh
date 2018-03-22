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

#ifndef __MATRIX_MATH_HXX__
#define __MATRIX_MATH_HXX__

template <int M, int K>
__device__ __inline__ void loadWeights(float weights_local[K], float* weights_remote, int layer, int row, int lda=M) {

    if (row >= M) return;

#pragma unroll
    for (int i=0; i<K; i++) {
        weights_local[i] = weights_remote[lda*K*layer + lda*i + row];
    }

}

// weights_unvectorized is an MxK matrix of col-major half values
// weights_vectorized is an MxK/2 matrix of half2 values, where each half2 contains adjacent entries of a single row but is otherwise still col-major 
// Total number of threads should equal M

static __device__ __forceinline__ half toHalf(float f) {
    return f;
}

static __device__ __forceinline__ half toHalf(half f) {
    return __float2half(f);
}

template <typename T>
__global__ void vectorizeWeights(int M, int K,half2* weights_vectorized, T* weights_unvectorized) {
    int row = blockIdx.x*blockDim.x + threadIdx.x;
    for (int k=0; k<K; k+= 2) { 
        half2 stage;
        stage.x = toHalf(weights_unvectorized[M*k + row]);
        stage.y = toHalf(weights_unvectorized[M*(k+1) + row]);
        weights_vectorized[M*(k/2) + row] = stage;
    }
}

template <int M, int K>
__device__ __inline__ void loadVectorizedWeights(half2 weights_local[K/2], half2* weights_remote, int layer, int thread_id, int lda=M) {

    //if (thread_id >= M) return;

    int row = thread_id;

#pragma unroll
    for (int i=0; i<K/2; i++) {
        weights_local[i] = weights_remote[lda*K/2*layer + lda*i + row];
    }
}


template <int K, int K_UNROLL, int TILE_N>
__device__ void GEMM(float weights[K], float activations[TILE_N][K], float accum[TILE_N]) {

    float accum_unrolled[TILE_N][K_UNROLL];

#pragma unroll
    for (int n=0; n<TILE_N; n++) {
#pragma unroll
        for (int u=0; u<K_UNROLL; u++) {
            accum_unrolled[n][u] = 0.f;
        }
    }

#pragma unroll
    for (int i=0; i<K; i += K_UNROLL) {
#pragma unroll
        for (int n=0; n<TILE_N; n++) {
#pragma unroll
            for (int u=0; u<K_UNROLL; u++) {
                accum_unrolled[n][u] += weights[i+u]*activations[n][i+u];
            }
        }
    }

#pragma unroll
    for (int n=0; n<TILE_N; n++) {
#pragma unroll
        for (int u=1; u<K_UNROLL; u++) {
            accum_unrolled[n][0] += accum_unrolled[n][u];
        }
    }

#pragma unroll
    for (int n=0; n<TILE_N; n++) {
        accum[n] = accum_unrolled[n][0];
    }

}

template <int K, int K_UNROLL, int TILE_N>
__device__ void GEMM(half2 weights[K/2], half activations[TILE_N][K], half accum[TILE_N]) {

    half2 accum2[TILE_N][K_UNROLL];

#pragma unroll
    for (int n=0; n<TILE_N; n++) {
#pragma unroll
        for (int u=0; u<K_UNROLL; u++) {
            accum2[n][u].x = 0.f;
            accum2[n][u].y = 0.f;
        }
    }

#pragma unroll
    for (int i=0; i<K/2; i+= K_UNROLL) {
#pragma unroll
        for (int n=0; n<TILE_N; n++) {
#pragma unroll
            for (int u=0; u<K_UNROLL; u++) {
                half2* activations2 = (half2*)activations[n];
                accum2[n][u] = __hfma2(weights[i+u], activations2[i+u], accum2[n][u]);        
            }
        }
    }

#pragma unroll
    for (int n=0; n<TILE_N; n++) {
#pragma unroll
        for (int u=1; u<K_UNROLL; u++) {
            accum2[n][0] = __hadd2(accum2[n][0], accum2[n][u]);
        }
    }

#pragma unroll
    for (int n=0; n<TILE_N; n++) {
        accum[n] = accum2[n][0].x + accum2[n][0].y;
    }
}

#endif
