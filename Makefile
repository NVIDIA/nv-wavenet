# ******************************************************************************
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of the NVIDIA CORPORATION nor the
#       names of its contributors may be used to endorse or promote products
#       derived from this software without specific prior written permission.
#     
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#  
# ******************************************************************************

NVCC = nvcc

ARCH=sm_61
NVCC_FLAGS = -arch=$(ARCH) -std=c++11 
NVCC_FLAGS += --use_fast_math

MAX_REGS = 128

HEADERS = nv_wavenet_util.cuh \
		  nv_wavenet_singleblock.cuh \
		  nv_wavenet_dualblock.cuh \
		  nv_wavenet_persistent.cuh \
		  nv_wavenet.cuh \
		  matrix_math.cuh \
		  softmax.cuh \
		  nv_wavenet_conversions.cuh

default: test

test : math_test nv_wavenet_test
	math_test
	nv_wavenet_test

nv_wavenet_perf : nv_wavenet_perf.cu $(HEADERS)
	$(NVCC) $(NVCC_FLAGS) -maxrregcount $(MAX_REGS) --ptxas-options=-v nv_wavenet_perf.cu -o nv_wavenet_perf

nv_wavenet_test : nv_wavenet_test.cu matrix.cpp matrix.h nv_wavenet_reference.cpp $(HEADERS)
	$(NVCC) $(NVCC_FLAGS) -lineinfo -maxrregcount $(MAX_REGS) nv_wavenet_test.cu matrix.cpp nv_wavenet_reference.cpp -o nv_wavenet_test

math_test : math_test.cu matrix_math.cuh matrix.cpp softmax.cuh
	$(NVCC) $(NVCC_FLAGS) math_test.cu matrix.cpp -lineinfo -o math_test

pytorch/_nv_wavenet_ext.so:
	$(MAKE) -C pytorch
	cd pytorch; python3 ./build.py

submodules:
	git submodule update --init

integration_test: submodules nv_wavenet_test pytorch/_nv_wavenet_ext.so
	./nv_wavenet_test
	cd pytorch; python3 ./integration_test.py

clean:
	rm  nv_wavenet_perf nv_wavenet_test math_test

