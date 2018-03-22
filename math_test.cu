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


#include <cublas_v2.h>
#include <assert.h>
#include <stdio.h>
#include <stdint.h>
#include "cuda_fp16.h"
#include "matrix.h"
#include "matrix_math.cuh"
#include "softmax.cuh"

#define gpuErrChk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

__global__ void convert_float2half_kernel(half* dst, float* src, size_t size) {

    int offset = blockDim.x*blockIdx.x + threadIdx.x;

    if (offset < size) {
        dst[offset] = __float2half(src[offset]);
    }
}

void convert_float2half(half* dst, float* src, size_t size) {
    float* tmp;
    gpuErrChk(cudaMallocHost(&tmp, size*sizeof(float)));
    memcpy(tmp, src, size*sizeof(float));

    dim3 block(256,1,1);
    dim3 grid((size + block.x - 1)/block.x, 1, 1);

    convert_float2half_kernel<<<grid,block>>>(dst, tmp, size);
    gpuErrChk(cudaDeviceSynchronize());

    gpuErrChk(cudaFreeHost(tmp));
}

__global__ void convert_half2float_kernel(float* dst, half* src, size_t size) {

    int offset = blockDim.x*blockIdx.x + threadIdx.x;

    if (offset < size) 
        dst[offset] = __half2float(src[offset]);
}

void convert_half2float(float* dst, half* dSrc, size_t size) {
    float* tmp;
    gpuErrChk(cudaMallocHost(&tmp, size*sizeof(float)));

    dim3 block(256,1,1);
    dim3 grid((size + block.x - 1)/block.x, 1, 1);

    convert_half2float_kernel<<<grid,block>>>(tmp, dSrc, size);
    gpuErrChk(cudaDeviceSynchronize());

    memcpy(dst, tmp, size*sizeof(float));

    gpuErrChk(cudaFreeHost(tmp));
}

template<typename T>
void allocate_buffers(int M, int N, int K, T*& A, T*& B, T*& C){
    gpuErrChk(cudaMalloc(&A, M*K*sizeof(T)));
    gpuErrChk(cudaMalloc(&B, K*N*sizeof(T)));
    gpuErrChk(cudaMalloc(&C, M*N*sizeof(T)));
    gpuErrChk(cudaMemset(A, 0, M*K*sizeof(T)));
    gpuErrChk(cudaMemset(B, 0, K*N*sizeof(T)));
    gpuErrChk(cudaMemset(C, 0, M*N*sizeof(T)));
}

template<typename T>
void clear_buffer(int M, int N, int K, T* C){
    gpuErrChk(cudaMemset(C, 0, M*N*sizeof(T)));
}

void upload_buffers(int M, int N, int K, float* dA, float* dB, float* hA, float* hB) {
    gpuErrChk(cudaMemcpy(dA, hA, M*K*sizeof(float), cudaMemcpyHostToDevice));
    gpuErrChk(cudaMemcpy(dB, hB, K*N*sizeof(float), cudaMemcpyHostToDevice));
}

void upload_buffers(int M, int N, int K, half* dA, half* dB, float* hA, float* hB) {
    convert_float2half(dA, hA, M*K);
    convert_float2half(dB, hB, K*N);
}

void download_buffer(int M, int N, int K, float* hC, float* dC) {
    gpuErrChk(cudaMemcpy(hC, dC, M*N*sizeof(float), cudaMemcpyDeviceToHost));
}

void download_buffer(int M, int N, int K, float* hC, half* dC) {
    convert_half2float(hC, dC, M*N);
}

template <typename T>
bool check_results(int M, int N, int K, Matrix& C, T* gpuC) {
    Matrix hGpuC(M,N,false);
    download_buffer(M,N,K,hGpuC.data(),gpuC);

    for (int i=0; i<M*N; i++) {
        if (C.data()[i] != hGpuC.data()[i]) {
            printf("mismatch %d %f %f\n", i, C.data()[i], hGpuC.data()[i]);
            return false; 
        }
        assert(C.data()[i] == hGpuC.data()[i]);
    }
    return true;
}

__device__ float toFloat(float f) { return f; }
__device__ float toFloat(half f) { return __half2float(f); } 

template<typename weights_T> constexpr __device__ int weightsK(int K) { return K; }
template<> constexpr __device__ int weightsK<half2>(int K) { return K/2; }

template<typename T_weight, typename T_data, int M, int K, int K_UNROLL, int TILE_N>
__device__ void gemm_kernel_inner(T_weight weights[weightsK<T_weight>(K)], int n_offset, T_data* B, T_data* C, int ldc) {
    __shared__ T_data activations[TILE_N][K];

    for (int n=n_offset; n<n_offset+TILE_N; n++) {
        for (int i=threadIdx.x; i<K; i+= blockDim.x) {
            activations[n-n_offset][i] = B[n*K + i];
        }
    }
    __syncthreads();

    T_data accum[TILE_N];

    GEMM<K,K_UNROLL,TILE_N>(weights,activations,accum);

    for (int n=n_offset; n<n_offset+TILE_N; n++) {
        C[n*ldc + threadIdx.x] = accum[n-n_offset];
    }

}

template <int M, int K>
__device__ __inline__ void loadWeights(half2 weights_local[K/2], half2* weights_remote, int layer, int row, int lda=M) {
    loadVectorizedWeights<M,K>(weights_local,weights_remote,layer,row,lda);
}

template<typename T_weight, typename T_data, int M, int K, int K_UNROLL>
__global__ void gemm_kernel(int N, T_weight* A, T_data* B, T_data* C, int lda=M, uint64_t* duration = NULL, int iterations_per_n = 1) {

    int ldc=lda;

    T_weight weights[weightsK<T_weight>(K)]; 
    loadWeights<M,K>(weights,A,0,threadIdx.x, lda);

    int tile_n = (N%2) == 0 ? 2 : 1;
    for (int n=0; n<N; n += tile_n) {
        if (tile_n == 2) {
            gemm_kernel_inner<T_weight,T_data,M,K,K_UNROLL,2>(weights, n, B, C, ldc);      
        }
        else {
            gemm_kernel_inner<T_weight,T_data,M,K,K_UNROLL,1>(weights, n, B, C, ldc);      
        }
    }

}

template<typename T_weight, typename T_data, int K_UNROLL>
void gemm(int M, int N, int K, T_weight* A, T_data* B, T_data* C, uint64_t* duration = NULL, int iterations_per_n = 1) {
    dim3 grid(1);
    dim3 block(M);
    if ((M==128) && (K==64)) {
        gemm_kernel<T_weight,T_data,128,64,K_UNROLL><<<grid,block>>>(N,A,B,C,M,duration,iterations_per_n);
    }
    else if ((M==128) && (K==256)) {
        gemm_kernel<T_weight,T_data,128,256,K_UNROLL><<<grid,block>>>(N,A,B,C,M,duration,iterations_per_n);
    }
    else if ((M==256) && (K==256)) {
        gemm_kernel<T_weight,T_data,256,256,K_UNROLL><<<grid,block>>>(N,A,B,C,M,duration,iterations_per_n);
    }
    else if ((M==32 && K==16)) {
        gemm_kernel<T_weight,T_data,32,16,K_UNROLL><<<grid,block>>>(N,A,B,C,M,duration,iterations_per_n);
    }
    else if ((M==16 && K==16)) {
        gemm_kernel<T_weight,T_data,16,16,K_UNROLL><<<grid,block>>>(N,A,B,C,M,duration,iterations_per_n);
    }
    else if ((M==16 && K==4)) {
        gemm_kernel<T_weight,T_data,16,4,K_UNROLL><<<grid,block>>>(N,A,B,C,M,duration,iterations_per_n);
    }
    else {
        assert(false);
    }
}

template<typename T_weight, typename T_data, int K_UNROLL, int TILE_M, int K>
void gemm_tiled(int M, int N, int K_deprecated, T_weight* A, T_data* B, T_data* C) {

    assert( (M%TILE_M) == 0);

    dim3 grid(1);
    dim3 block(TILE_M);

    assert(M == TILE_M*2);

    // Bottom half
    gemm_kernel<T_weight,T_data,TILE_M,K,K_UNROLL><<<grid,block>>>(N,A,B,C,M,NULL,1);
    // Top half
    gemm_kernel<T_weight,T_data,TILE_M,K,K_UNROLL><<<grid,block>>>(N,A+TILE_M,B,C+TILE_M,M,NULL,1);
}   


template <typename T, int NUM_THREADS, int NUM_ROWS, int UNROLL>
__global__ void softmax_multi_kernel(int N, T* input, T* output, float* selector, int* selection) {
    softmax_select<T,NUM_THREADS,NUM_ROWS,UNROLL>(0,N,input,output,selector,selection,0,blockDim.x);
}

template<typename T>
void softmax_multi_select(int M, int N, T* in, T* out, float* selector, int* selection) {

    dim3 grid(1);

    if ((N%2) == 0) {
        if (M==256) softmax_multi_kernel<T,128,256,2><<<grid,128>>>(N,in, out, selector, selection);
        else if (M==128) softmax_multi_kernel<T,64,128,2><<<grid,64>>>(N,in,out,selector,selection);
        else if (M==32) softmax_multi_kernel<T,32,32,2><<<grid,32>>>(N,in,out,selector,selection);
        else assert(false);
    }
    else {
        if (M==256) softmax_multi_kernel<T,128,256,1><<<grid,128>>>(N,in, out, selector, selection);
        else if (M==128) softmax_multi_kernel<T,64,128,1><<<grid,64>>>(N,in,out,selector,selection);
        else if (M==32) softmax_multi_kernel<T,32,32,1><<<grid,32>>>(N,in,out,selector,selection);
        else assert(false);
    }

}


int main(void) {

    /*
       const int M = 128;
       const int N = 1;
       const int K = 64;
     */
    /*
       const int M = 16;
       const int N = 1;
       const int K = 4;
     */
    const int M = 128;
    const int N = 16;
    const int K = 64;


    Matrix A(M,K,false);
    Matrix B(K,N,false);
    Matrix C(M,N,false);

    for (int col=0; col<K; col++) {
        for (int row=0; row<M; row++) {
            A.set(row, col, ((row+1) + (col+1)) % 6 );
        }
    }

    for (int row=0; row<K; row++) {
        for (int col=0; col<N; col++) {
            B.set(row, col, ((row+2) + (col+1)) % 5 );
        }
    }

    for (int m=0; m<M; m++) {
        for (int n=0; n<N; n++) {
            C.set(m,n,0.f);
        }
    }

    matrix_multiply(C,A,B);

    printf("Testing GEMM - FP32\n");
    float* gpuA;
    float* gpuB;
    float* gpuC;

    allocate_buffers<float>(M,N,K,gpuA,gpuB,gpuC);
    upload_buffers(M,N,K,gpuA, gpuB, A.data(), B.data());

    //printf("K_unroll = 1\n");
    clear_buffer(M,N,K,gpuC);
    gemm<float,float,1>(M,N,K,gpuA,gpuB,gpuC);
    assert(check_results<float>(M,N,K,C,gpuC));

    //printf("K_unroll = 2\n");
    clear_buffer(M,N,K,gpuC);
    gemm<float,float,2>(M,N,K,gpuA,gpuB,gpuC);
    assert(check_results<float>(M,N,K,C,gpuC));

    //printf("Tiled\n");
    clear_buffer(M,N,K,gpuC);
    gemm_tiled<float,float,1,M/2,K>(M,N,K,gpuA,gpuB,gpuC);
    assert(check_results<float>(M,N,K,C,gpuC));

    printf("Testing GEMM - FP16\n");
    half* gpuA_half;
    half* gpuB_half;
    half* gpuC_half;

    allocate_buffers<half>(M,N,K,gpuA_half,gpuB_half,gpuC_half);
    upload_buffers(M,N,K,gpuA_half, gpuB_half, A.data(), B.data());

    half2* gpuA_vectorized;
    gpuErrChk(cudaMalloc(&gpuA_vectorized, M*K/2*sizeof(half2)));
    dim3 block(128);
    dim3 grid((M + block.x - 1)/block.x);
    vectorizeWeights<half><<<grid,block>>>(M,K, gpuA_vectorized, gpuA_half);
    gpuErrChk(cudaDeviceSynchronize());

    clear_buffer(M,N,K,gpuC_half);
    gemm<half2,half,1>(M,N,K,gpuA_vectorized,gpuB_half,gpuC_half);
    assert(check_results<half>(M,N,K,C,gpuC_half));

    clear_buffer(M,N,K,gpuC_half);
    gemm<half2,half,2>(M,N,K,gpuA_vectorized,gpuB_half,gpuC_half);
    assert(check_results<half>(M,N,K,C,gpuC_half));

    clear_buffer(M,N,K,gpuC_half);
    gemm_tiled<half2,half,1,M/2,K>(M,N,K,gpuA_vectorized,gpuB_half,gpuC_half);
    assert(check_results<half>(M,N,K,C,gpuC_half));


    printf("Testing softmax - FP32 \n");

    for (int r=0; r<M; r++) {
        for (int c=0; c<N; c++) {
            C.set(r,c,tanh(float(r+c)/M)); 
        }
    }
    gpuErrChk(cudaMemcpy(gpuC, C.data(), M*N*sizeof(float), cudaMemcpyHostToDevice));

    float* gpu_softmax_out_float;
    gpuErrChk(cudaMalloc(&gpu_softmax_out_float, M*N*sizeof(float)));

    Matrix ref_softmax_out(M,N,false);
    matrix_softmax(ref_softmax_out, C);

    Matrix hGpuC(M,N,false);

    gpuErrChk(cudaMemset(gpu_softmax_out_float, 0, M*N*sizeof(float)));

    float* selector;
    gpuErrChk(cudaHostAlloc(&selector,N*sizeof(float),cudaHostAllocMapped));

    int* selection;
    gpuErrChk(cudaHostAlloc(&selection,N*sizeof(int),cudaHostAllocMapped));
    for (int i=0; i<N; i++) selection[i] = -1;

    for (int i=0; i<N; i++) {
        selector[i] = float(i)/N;
    }

    gpuErrChk(cudaMemset(gpu_softmax_out_float, 0, M*N*sizeof(float)));
    for (int i=0; i<N; i++) selection[i] = -1;
    softmax_multi_select<float>(M,N,gpuC,gpu_softmax_out_float,selector,selection);
    gpuErrChk(cudaDeviceSynchronize());

    download_buffer(M,N,K,hGpuC.data(),gpu_softmax_out_float);
    matrix_compare("softmax_out", ref_softmax_out, hGpuC, 1.e-3);

    for (int i=0; i<N; i++) {
        float ref_sel = 0.f;
        for (int r=0; r<M; r++) {
            float ref_sel_next = ref_sel + ref_softmax_out.get(r,i);
            if (selector[i] < ref_sel_next) {
                assert(r == selection[i]);
                break;
            }
        }
    }

    gpuErrChk(cudaMemset(gpu_softmax_out_float, 0, M*N*sizeof(float)));
    printf("All done!\n");




    return 0;
}
