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

__global__ void convert_float2half_kernel(half* dst, float* src, size_t size) {

    int totalThreads = gridDim.x * blockDim.x;
    int blockStart = blockDim.x * blockIdx.x;

    for (int i = blockStart + threadIdx.x; i < size; i += totalThreads) {
        dst[i] = __float2half(src[i]);
    }
}

bool isDevicePtr(const void* ptr) {
    cudaPointerAttributes attributes;
    cudaError_t result = cudaPointerGetAttributes(&attributes, ptr);
    return (result == cudaSuccess) && (attributes.memoryType == cudaMemoryTypeDevice);
}


void convert_float2half(half* dst, float* src, size_t size) {
    float* tmp;

    if (isDevicePtr(src)) {
        tmp = src;
    }
    else {
        gpuErrChk(cudaMallocHost(&tmp, size*sizeof(float)));
        memcpy(tmp, src, size*sizeof(float));
    }

    dim3 block(256,1,1);
    dim3 grid(256,1,1);

    convert_float2half_kernel<<<grid,block>>>(dst, tmp, size);
    gpuErrChk(cudaDeviceSynchronize());

    if (!isDevicePtr(src)) gpuErrChk(cudaFreeHost(tmp));
}


void convert_float2half2_vectorized(half2* dst, float* src, int M, int K) {
    float* tmp;

    if (isDevicePtr(src)) {
        tmp = src;
    }
    else {
        gpuErrChk(cudaMallocHost(&tmp, M*K*sizeof(float)));
        memcpy(tmp, src, M*K*sizeof(float));
    }

    dim3 block(32);
    assert((M%block.x)==0);
    dim3 grid(M/block.x);

    vectorizeWeights<float><<<grid,block>>>(M,K,dst, tmp);
    gpuErrChk(cudaDeviceSynchronize());

    gpuErrChk(cudaDeviceSynchronize());
    if (!isDevicePtr(src)) gpuErrChk(cudaFreeHost(tmp));
}

__global__ void convert_half2float_kernel(float* dst, half* src, size_t size) {

    int offset = blockDim.x*blockIdx.x + threadIdx.x;

    if (offset < size) 
        dst[offset] = __half2float(src[offset]);
}

void convert_half2float(float* dst, half* dSrc, size_t size) {
    float* tmp;

    if (isDevicePtr(dst)) {
        tmp = dst;
    }
    else {
        gpuErrChk(cudaMallocHost(&tmp, size*sizeof(float)));
    }

    dim3 block(256,1,1);
    dim3 grid((size + block.x - 1)/block.x, 1, 1);

    convert_half2float_kernel<<<grid,block>>>(tmp, dSrc, size);
    gpuErrChk(cudaDeviceSynchronize());

    if (!isDevicePtr(dst)) {
        memcpy(dst, tmp, size*sizeof(float));
        gpuErrChk(cudaFreeHost(tmp));
    }
}
