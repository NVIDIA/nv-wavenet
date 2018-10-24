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

#include "matrix.h"
#include "nv_wavenet.cuh"
#include "nv_wavenet_util.cuh"
#include "nv_wavenet_reference.h"
#include <assert.h>
#include <stdio.h>
#include <vector>

Matrix* createMatrix(int r, int c) {
    float mean = 0.0;
    float scale = 0.5 / r;
    Matrix* m = new Matrix(r,c,false);
    m->randomize(mean,scale);
   return m;
}

template <typename T_weight, typename T_data, int R, int S, int A>
void runTest(int num_layers, int max_dilation, int batch_size, int num_iterations, int samples_per_iteration, int impl, bool inputsFromDevice=false, bool weightsFromDevice=false) {

    float mean = 0.0;
    float scale = 0.5 / R;

    // Just encode one-hot vector as an integer
    std::vector<int> yInPrev(batch_size);
    std::vector<int> yInCur(batch_size);

    for (int b=0; b<batch_size; b++) {
        yInPrev[b] = rand() % A;
        yInCur[b] = rand() % A;
    }
    std::vector<int> yOut(batch_size);

    Matrix outputSelectors(batch_size,samples_per_iteration);
    outputSelectors.randomize(0.5,1.0);

    Matrix embeddingsPrev(R,A,false);
    Matrix embeddingsCur(R,A,false);

    embeddingsPrev.randomize(mean,scale);
    embeddingsCur.randomize(mean,scale);

    std::vector<Matrix*> Wprev(num_layers);
    std::vector<Matrix*> Wcur(num_layers);
    std::vector<Matrix*> Bh(num_layers);
    std::vector<Matrix*> Wres(num_layers);
    std::vector<Matrix*> Bres(num_layers);
    std::vector<Matrix*> Wskip(num_layers);
    std::vector<Matrix*> Bskip(num_layers);
    std::vector<Matrix*> skipOut(num_layers+1);

    // Retain results for dilated inputs
    std::vector<std::vector<Matrix*>> Xt(samples_per_iteration);
    for (int sample=0; sample<samples_per_iteration; sample++) {
        Xt[sample].resize(num_layers+1);
    }

    for (int l=0; l<num_layers; l++) {
        // Weights
        Wprev[l] = createMatrix(2*R,R);
        Wcur[l] = createMatrix(2*R,R);
        Bh[l] = createMatrix(2*R,1);
        Wres[l] = createMatrix(R,R);
        Bres[l] = createMatrix(R,1);
        Wskip[l] = createMatrix(S,R);
        Bskip[l] = createMatrix(S,1);

        // Activations
        skipOut[l] = createMatrix(S,batch_size);
    }

    for (int sample=0; sample<samples_per_iteration; sample++) {
        for (int layer=0; layer<num_layers+1; layer++) {
            Xt[sample][layer] = createMatrix(R, batch_size);
        }
    }

    Matrix WskipOut(A,S,false);
    WskipOut.randomize(mean,scale);
    Matrix BskipOut(A,1,false);
    BskipOut.randomize(mean, scale);
    Matrix Wout(A,A,false);
    Wout.randomize(mean,scale);
    Matrix Bout(A,1,false);
    Bout.randomize(mean,scale);

    Matrix skipOutFinal(A,batch_size,false);
    Matrix out(A,batch_size,false);
    Matrix p(A,batch_size,false);

    Matrix zero(S,batch_size,false);
    for (int row = 0; row < S; row++) {
        for (int col = 0; col < batch_size; col++) {
            zero.set(row,col,0.f);
        }
    }

    nvWavenetReference ref(num_layers, batch_size, samples_per_iteration, R, S, A, max_dilation);
    nvWavenetInfer<T_weight,T_data,R,S,A>* infer = new nvWavenetInfer<T_weight,T_data,R,S,A>(num_layers, max_dilation, batch_size, samples_per_iteration, impl);

    ref.setEmbeddings(embeddingsPrev.data(), embeddingsCur.data());
    for (int l=0; l<num_layers; l++) {
        ref.setLayerWeights(l, Wprev[l]->data(), Wcur[l]->data(), Bh[l]->data(), Wres[l]->data(), Bres[l]->data(), Wskip[l]->data(), Bskip[l]->data());
    }
    ref.setOutWeights(WskipOut.data(), BskipOut.data(), Wout.data(), Bout.data());

    if (weightsFromDevice) {
        float* d_embeddingsPrev;
        float* d_embeddingsCur;
        gpuErrChk(cudaMalloc(&d_embeddingsPrev, R*A*sizeof(float)));
        gpuErrChk(cudaMemcpy(d_embeddingsPrev, embeddingsPrev.data(), R*A*sizeof(float), cudaMemcpyHostToDevice));
        gpuErrChk(cudaMalloc(&d_embeddingsCur, R*A*sizeof(float)));
        gpuErrChk(cudaMemcpy(d_embeddingsCur, embeddingsCur.data(), R*A*sizeof(float), cudaMemcpyHostToDevice));

        infer->setEmbeddings(d_embeddingsPrev, d_embeddingsCur);

        gpuErrChk(cudaFree(d_embeddingsPrev));
        gpuErrChk(cudaFree(d_embeddingsCur));

        float* d_Wprev;
        float* d_Wcur;
        float* d_Bh;
        float* d_Wres;
        float* d_Bres;
        float* d_Wskip;
        float* d_Bskip;
        for (int l=0; l<num_layers; l++) {
            gpuErrChk(cudaMalloc(&d_Wprev, 2*R*R*sizeof(float)));
            gpuErrChk(cudaMemcpy(d_Wprev, Wprev[l]->data(), 2*R*R*sizeof(float), cudaMemcpyHostToDevice));
            gpuErrChk(cudaMalloc(&d_Wcur, 2*R*R*sizeof(float)));
            gpuErrChk(cudaMemcpy(d_Wcur, Wcur[l]->data(), 2*R*R*sizeof(float), cudaMemcpyHostToDevice));
            gpuErrChk(cudaMalloc(&d_Bh, 2*R*sizeof(float)));
            gpuErrChk(cudaMemcpy(d_Bh, Bh[l]->data(), 2*R*sizeof(float), cudaMemcpyHostToDevice));
            gpuErrChk(cudaMalloc(&d_Wres, R*R*sizeof(float)));
            gpuErrChk(cudaMemcpy(d_Wres, Wres[l]->data(), R*R*sizeof(float), cudaMemcpyHostToDevice));
            gpuErrChk(cudaMalloc(&d_Bres, R*sizeof(float)));
            gpuErrChk(cudaMemcpy(d_Bres, Bres[l]->data(), R*sizeof(float), cudaMemcpyHostToDevice));
            gpuErrChk(cudaMalloc(&d_Wskip, S*R*sizeof(float)));
            gpuErrChk(cudaMemcpy(d_Wskip, Wskip[l]->data(), S*R*sizeof(float), cudaMemcpyHostToDevice));
            gpuErrChk(cudaMalloc(&d_Bskip, S*sizeof(float)));
            gpuErrChk(cudaMemcpy(d_Bskip, Bskip[l]->data(), S*sizeof(float), cudaMemcpyHostToDevice));

            infer->setLayerWeights(l, d_Wprev, d_Wcur, d_Bh, d_Wres, d_Bres, d_Wskip, d_Bskip);

            gpuErrChk(cudaFree(d_Wprev));
            gpuErrChk(cudaFree(d_Wcur));
            gpuErrChk(cudaFree(d_Bh));
            gpuErrChk(cudaFree(d_Wres));
            gpuErrChk(cudaFree(d_Bres));
            gpuErrChk(cudaFree(d_Wskip));
            gpuErrChk(cudaFree(d_Bskip));
        }

        float* d_WskipOut;
        float* d_BskipOut;
        float* d_Wout;
        float* d_Bout;

        gpuErrChk(cudaMalloc(&d_WskipOut, A*S*sizeof(float)));
        gpuErrChk(cudaMemcpy(d_WskipOut, WskipOut.data(), A*S*sizeof(float), cudaMemcpyHostToDevice));
        gpuErrChk(cudaMalloc(&d_BskipOut, A*sizeof(float)));
        gpuErrChk(cudaMemcpy(d_BskipOut, BskipOut.data(), A*sizeof(float), cudaMemcpyHostToDevice));
        gpuErrChk(cudaMalloc(&d_Wout, A*A*sizeof(float)));
        gpuErrChk(cudaMemcpy(d_Wout, Wout.data(), A*A*sizeof(float), cudaMemcpyHostToDevice));
        gpuErrChk(cudaMalloc(&d_Bout, A*sizeof(float)));
        gpuErrChk(cudaMemcpy(d_Bout, Bout.data(), A*sizeof(float), cudaMemcpyHostToDevice));
        
        infer->setOutWeights(d_WskipOut, d_BskipOut, d_Wout, d_Bout);

        gpuErrChk(cudaFree(d_WskipOut));
        gpuErrChk(cudaFree(d_BskipOut));
        gpuErrChk(cudaFree(d_Wout));
        gpuErrChk(cudaFree(d_Bout));
        
    }
    else {
        infer->setEmbeddings(embeddingsPrev.data(), embeddingsCur.data());
        for (int l=0; l<num_layers; l++) {
            infer->setLayerWeights(l, Wprev[l]->data(), Wcur[l]->data(), Bh[l]->data(), Wres[l]->data(), Bres[l]->data(), Wskip[l]->data(), Bskip[l]->data());
        }
        infer->setOutWeights(WskipOut.data(), BskipOut.data(), Wout.data(), Bout.data());
    }

    Matrix zeroMatrix(R,batch_size,false);
    for (int row=0; row<R; row++) {
        for (int col=0; col<batch_size; col++) {
            zeroMatrix.set(row,col,0.f);
        }
    }

    Matrix Lh(2*R,samples_per_iteration*num_layers*batch_size);
    assert(Lh.data());
    Lh.randomize(mean,scale);

    ref.setInputs(Lh.data(), outputSelectors.data());

    if (inputsFromDevice) {
        float* d_Lh;
        gpuErrChk(cudaMalloc(&d_Lh, 2*R*samples_per_iteration*num_layers*batch_size*sizeof(float)));
        float* d_outputSelectors;
        gpuErrChk(cudaMalloc(&d_outputSelectors,samples_per_iteration*batch_size*sizeof(float)));

        gpuErrChk(cudaMemcpy(d_Lh, Lh.data(), 2*R*samples_per_iteration*num_layers*batch_size*sizeof(float), cudaMemcpyHostToDevice));
        gpuErrChk(cudaMemcpy(d_outputSelectors, outputSelectors.data(), samples_per_iteration*batch_size*sizeof(float), cudaMemcpyHostToDevice));

        infer->setInputs(d_Lh, d_outputSelectors);

        gpuErrChk(cudaFree(d_Lh));
        gpuErrChk(cudaFree(d_outputSelectors));
    }
    else {
        infer->setInputs(Lh.data(), outputSelectors.data());
    }

    for (int i=0; i<num_iterations; i++) {
        printf("Iteration: %d\n", i);

        // Run reference implementation


        int batch_size_per_block = ((batch_size % 4) == 0) ? 4 : ((batch_size % 2) == 0) ? 2 : 1;

        int* refYout = (int*)malloc(samples_per_iteration*batch_size*sizeof(int));
        int* mcYout = (int*)malloc(samples_per_iteration*batch_size*sizeof(int));

        ref.run(samples_per_iteration, batch_size, refYout);

        assert(infer->run_chunks(7, [](int*, int, int){}, samples_per_iteration, batch_size, mcYout, batch_size_per_block));
        gpuErrChk(cudaDeviceSynchronize());

        // Check results

        for (int l=0; l<num_layers; l++) {

            printf("Checking layer %d\n", l);

            Matrix refXout(R,batch_size);
            Matrix refSkipOut(S, batch_size);
            ref.getXtOut(l, refXout.data());
            ref.getSkipOut(l, refSkipOut.data());

            Matrix mcXout(R,batch_size,false);
            Matrix mcSkipOut(S,batch_size,false);
            infer->getXtOut(l, mcXout.data());
            infer->getSkipOut(l, mcSkipOut.data());

            matrix_compare("Xout", refXout, mcXout, 1.e-2);
            matrix_compare("skipOut", refSkipOut, mcSkipOut, 1.e-2, true);
        }

        Matrix refSkipOutFinal(A,batch_size);
        ref.getZs(refSkipOutFinal.data());

        Matrix mcSkipOutFinal(A,batch_size,false);
        infer->getZs(mcSkipOutFinal.data());

        matrix_compare("Zs", refSkipOutFinal, mcSkipOutFinal, 1.e-4, true);

        Matrix refOut(A,batch_size);
        ref.getZa(refOut.data());

        Matrix mcOut(A,batch_size,false);
        infer->getZa(mcOut.data());

        matrix_compare("Za", refOut, mcOut, 1.e-4);

        Matrix refP(A,batch_size);
        ref.getP(refP.data());

        Matrix mcP(A,batch_size,false);
        infer->getP(mcP.data());
        matrix_compare("p",refP,mcP,1.e-3);

        printf("Comparing yOut\n");

        for (int i=0; i<samples_per_iteration*batch_size; i++) {
            assert(refYout[i] == mcYout[i]);
        }
        free(mcYout);
        free(refYout);

        printf("SUCCESS!\n");
    }


    // Clean up

    delete infer;

    for (int l=0; l<num_layers; l++) {
        delete Wprev[l];
        delete Wcur[l];
        delete Bh[l];
        delete Wres[l];
        delete Bres[l];
        delete Wskip[l];
        delete Bskip[l];
        for (int sample=0; sample<samples_per_iteration;sample++) {
            delete Xt[sample][l];
        }
        delete skipOut[l];
    }
}

int main(int argc, char* argv[]) {

    int num_layers = 20;
    int batch_size = 16;

    if (argc > 1) num_layers = atoi(argv[1]);
    if (argc > 2) batch_size  = atoi(argv[2]);

    // How many samples to generate each time we invoke the kernel
    const int SAMPLES_PER_ITERATION = 8;
    const int MAX_DILATION = SAMPLES_PER_ITERATION;

    srand(3);

    printf("Testing R=32, S=128\n");
    printf("   Testing Single-Block\n");
    runTest<float,float,32,128, 256>(num_layers, MAX_DILATION, batch_size, 2, SAMPLES_PER_ITERATION, 1);
    printf("   Testing Dual-Block\n");
    runTest<float,float,32,128, 256>(num_layers, MAX_DILATION, batch_size, 2, SAMPLES_PER_ITERATION, 2);
    printf("   Testing Persistent\n");
    runTest<float,float,32,128, 256>(num_layers, MAX_DILATION, batch_size, 2, SAMPLES_PER_ITERATION, 3);
    printf("   Testing Manyblock\n");
    runTest<float,float,32,128, 256>(num_layers, MAX_DILATION, batch_size, 2, SAMPLES_PER_ITERATION, 4);

    srand(10);

    printf("Testing R=64, S=128\n");
    printf("   Testing Single-Block\n");
    runTest<float,float,64,128, 256>(num_layers, MAX_DILATION, batch_size, 2, SAMPLES_PER_ITERATION, 1, true, false);
    printf("   Testing Dual-Block\n");
    runTest<float,float,64,128, 256>(num_layers, MAX_DILATION, batch_size, 2, SAMPLES_PER_ITERATION, 2, false, true);
    printf("   Testing Persistent\n");
    runTest<float,float,64,128, 256>(num_layers, MAX_DILATION, batch_size, 2, SAMPLES_PER_ITERATION, 3, true, true);
    printf("   Testing Manyblock\n");
    runTest<float,float,64,128, 256>(num_layers, MAX_DILATION, batch_size, 2, SAMPLES_PER_ITERATION, 4, true, true);

    srand(30);

    printf("Testing R=64, S=256\n");
    printf("    Testing Single-Block\n");
    runTest<float,float,64,256, 256>(num_layers, MAX_DILATION, batch_size, 2, SAMPLES_PER_ITERATION, 1);
    printf("    Testing Dual-Block\n");
    runTest<float,float,64,256, 256>(num_layers, MAX_DILATION, batch_size, 2, SAMPLES_PER_ITERATION, 2);
    printf("    Testing Persistent\n");
    runTest<float,float,64,256, 256>(num_layers, MAX_DILATION, batch_size, 2, SAMPLES_PER_ITERATION, 3);
    printf("   Testing Manyblock\n");
    runTest<float,float,64,256, 256>(num_layers, MAX_DILATION, batch_size, 2, SAMPLES_PER_ITERATION, 4);

    srand(50);

    printf("Testing R=128, S=256\n");
    printf("    Testing Persistent\n");
    runTest<float,float,128,256, 256>(num_layers, MAX_DILATION, batch_size, 2, SAMPLES_PER_ITERATION, 3);
    printf("   Testing Manyblock\n");
    runTest<float,float,128,256, 256>(num_layers, MAX_DILATION, batch_size, 2, SAMPLES_PER_ITERATION, 4);

    srand(70);

    printf("Testing A=512\n");
    printf("    Testing Persistent\n");
    runTest<float,float,64,128, 512>(num_layers, MAX_DILATION, batch_size, 2, SAMPLES_PER_ITERATION, 3);
    printf("Testing A=1024\n");
    runTest<float,float,128,256, 1024>(12, MAX_DILATION, batch_size, 2, SAMPLES_PER_ITERATION, 3);
}
