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

#ifndef __SOFTMAX_HXX__
#define __SOFTMAX_HXX__


__device__ __forceinline__ void namedBarrierSync(int name, int numThreads) {
      asm volatile("bar.sync %0, %1;" : : "r"(name), "r"(numThreads) : "memory");
}

template <typename T, int NUM_THREADS, int NUM_ROWS, int UNROLL>
__device__ __inline__ void softmax_select(int first_col, int num_cols, T* input, T* output, float* selector, int* selection, int barrierName, int numThreads, int thread_id = threadIdx.x) {

    const int ROWS_PER_THREAD = NUM_ROWS / NUM_THREADS;

    for (int col=first_col; col<first_col+num_cols; col+=UNROLL) {
        // Load activations
        float activations_in[UNROLL][ROWS_PER_THREAD];
        for (int u=0; u<UNROLL; u++) {
            for (int r=0; r<ROWS_PER_THREAD; r++) {
                activations_in[u][r] = input[(col+u)*NUM_ROWS + thread_id*ROWS_PER_THREAD + r];
            }
        }

        float local_max[UNROLL];
        float local_sum[UNROLL];
        for (int u=0; u<UNROLL; u++) {
            local_max[u] = activations_in[u][0];
            local_sum[u] = 0.f;
        }

        // Compute the max first so we can subtract it from the inputs and prevent explosions
#pragma unroll
        for (int u=0; u<UNROLL; u++) {
#pragma unroll
            for (int r=1; r<ROWS_PER_THREAD; r++) {
                if (activations_in[u][r] > local_max[u])  local_max[u] = activations_in[u][r];
            }
        }

        // Each warp computes its max
        __shared__ float warp_max[UNROLL][NUM_THREADS/32];
#pragma unroll
        for (int u=0; u<UNROLL; u++) {
            float wmax = local_max[u];
            for (int offset=16; offset>0; offset /= 2) {
                float v = __shfl_down_sync(0xFFFFFFFF,wmax,offset);
                if (v > wmax) wmax = v;
            }
            if ((thread_id%32)==0) {
                warp_max[u][thread_id/32] = wmax;
            }
        }

        namedBarrierSync(barrierName, numThreads);

        // Now each thread computes max across warps
#pragma unroll
        for (int u=0; u<UNROLL; u++) {
#pragma unroll
            for (int w=0; w<NUM_THREADS/32; w++) {
                float wmax = warp_max[u][w];
                if (wmax > local_max[u]) local_max[u] = wmax;
            }
        }

        // Subtract the max from the input
#pragma unroll
        for (int u=0; u<UNROLL; u++) {
#pragma unroll
            for (int r=0; r<ROWS_PER_THREAD; r++) {
                activations_in[u][r] -= local_max[u];
            }
        }
       
        // Exponentiate and sum
#pragma unroll
        for (int u=0; u<UNROLL; u++) {
#pragma unroll
            for (int r=0; r<ROWS_PER_THREAD; r++) {
                float act_exp = expf(activations_in[u][r]);
                activations_in[u][r] = act_exp;
                local_sum[u] += activations_in[u][r];
            }
        }

        __shared__ float thread_sums[UNROLL][NUM_ROWS];
        __shared__ float warp_sums[UNROLL][NUM_THREADS/32];

        for (int u=0; u<UNROLL; u++) {
            thread_sums[u][thread_id] = local_sum[u];
        }

#pragma unroll
        for (int u=0; u<UNROLL; u++) {
            float accum = local_sum[u];
            for (int offset=16; offset>0; offset /= 2) {
                accum += __shfl_down_sync(0xFFFFFFFF,accum,offset);
            }
            if ((thread_id&0x1f)==0) {
                warp_sums[u][thread_id/32] = accum;
            }
        }
        namedBarrierSync(barrierName, numThreads);

        float sum[UNROLL];
#pragma unroll
        for (int u=0; u<UNROLL; u++) {
            sum[u] = 0.f;
            for (int w=0; w<NUM_THREADS/32;w++) {
                sum[u] += warp_sums[u][w];
            }
        }

        if (output != NULL) {
#pragma unroll
            for (int u=0; u<UNROLL; u++) {
                for (int r=0; r<ROWS_PER_THREAD; r++) {
                    int row = thread_id*ROWS_PER_THREAD+r;
                    output[(col+u)*NUM_ROWS + row] = activations_in[u][r] / sum[u];
                }
            }
        }

        for (int u=0; u<UNROLL; u++) {
            // Now do the weighted selection
            float sel = selector[col+u] * sum[u];  
            float wsum = 0.f;
            // Write out a value in case scan fails
            selection[col+u] = 128;
            namedBarrierSync(barrierName, numThreads);
            int warp_id = thread_id/32;
            for (int w=0; w<NUM_THREADS/32; w++) {
                float wsum_next = wsum + warp_sums[u][w];
                if (sel <= wsum_next) {
                    if ( warp_id == w) {
                        // We're in the right warp
                        float tsum = wsum; 
                        for (int i=0; i<32; i++) {
                            float tsum_next = tsum + thread_sums[u][w*32+i];
                            if (sel <= tsum_next) {
                                if ((thread_id&0x1f) == i) {
                                    // We're in the right thread
                                    float s = tsum;
                                    for (int r=0; r<ROWS_PER_THREAD; r++) {
                                        float s_next = s + activations_in[u][r];
                                        if (sel <= s_next) {
                                            // Found it!
                                            selection[col+u] = thread_id*ROWS_PER_THREAD + r;
                                            break;
                                        }
                                        s = s_next;
                                    }
                                }
                                break;
                            }
                            tsum = tsum_next;
                        }
                    }
                    break;
                }
                wsum = wsum_next;
            }
        }
    }
}

#endif
