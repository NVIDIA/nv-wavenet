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

#ifndef __DEEPVOICE_UTIL_H__
#define __DEEPVOICE_UTIL_H__

#include <stdio.h>
#include "cuda_occupancy.h"

#define gpuErrChk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

int getOccupancy(int deviceId, size_t blockSize, void* func) {
    cudaDeviceProp prop;
    gpuErrChk ( cudaGetDeviceProperties(&prop, 0) );
    cudaOccDeviceProp occProp = prop;

    cudaFuncAttributes attr;
    gpuErrChk ( cudaFuncGetAttributes(&attr, func) );
    cudaOccFuncAttributes occAttr = attr;

    cudaOccDeviceState occState;

    cudaOccResult result;
    cudaOccMaxActiveBlocksPerMultiprocessor(&result, &occProp, &occAttr, &occState, blockSize, 0);

    return result.activeBlocksPerMultiprocessor;

}

__device__ __forceinline__ half loadVolatile(const volatile half* y, int index) {
    const volatile __half_raw* chr = (reinterpret_cast<const volatile __half_raw *>(y) );
    __half_raw hr;
    hr.x = chr[index].x;
    return half( hr );
}
__device__ __forceinline__ void storeVolatile(volatile half* y, int index, half val) {
    half* y_nv = (half*)y;
    y_nv[index] = val;
}

__device__ __forceinline__ float loadVolatile(const volatile float* y, int index) {
    return y[index];
}
__device__ __forceinline__ void storeVolatile(volatile float* y, int index, float val) {
    y[index] = val;
}

__forceinline__ __device__ float sigmoid(float in) {
    float ans = 1.f / (1.f + expf(-in));
    return ans;
}

__forceinline__ __device__ float _tanh(float in) {
    float ans = tanhf(in);
    return ans;
}

__device__ __forceinline__ float relu(float f) { return (f < 0.f) ? 0.f : f; }
__device__ __forceinline__ half relu(half h) { half zero = 0.f; return (h < zero) ? zero : h; }

#endif
