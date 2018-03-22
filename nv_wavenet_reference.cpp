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
#include <math.h>
#include <assert.h>
#include <vector>
#include <stdio.h>
#include <string.h>

#include "nv_wavenet_reference.h"

float sigmoid(float f) { return 1.f / (1.f + exp(-f)); }

float tanh_proxy(float f) { return tanh(f); }
float sigmoid_proxy(float f) { return sigmoid(f); }

void nvWavenetEmbed(std::vector<int>& yInPrev, std::vector<int>& yInCur, Matrix& embeddingsPrev, Matrix& embeddingsCur, Matrix& x0){
    assert(yInPrev.size() == x0.cols());
    assert(yInCur.size() == x0.cols());
    assert(embeddingsPrev.rows() == x0.rows());
    assert(embeddingsCur.rows() == x0.rows());

    for (int batch_id=0; batch_id<yInPrev.size(); batch_id++) {
        float prev = yInPrev[batch_id]; 
        float cur = yInCur[batch_id]; 
        for (int r=0;r<embeddingsPrev.rows();r++){
            float embedded = tanh(embeddingsPrev.get(r, prev) + embeddingsCur.get(r, cur));
            x0.set(r,batch_id,embedded);
        }
    }
}

void nvWavenetLayer(int r, int batch_size, Matrix& Wprev, Matrix& Wcur, Matrix& Bh, Matrix& Lh, Matrix& Wres, Matrix& Bres, Matrix& Wskip, Matrix& Bskip, Matrix& Xtmd, Matrix& Xin, Matrix& Xout, Matrix& skipIn, Matrix& skipOut, bool lastLayer) {

    Matrix a_prev(2*r, batch_size, false);
    matrix_multiply(a_prev, Wprev, Xtmd);

    Matrix a_cur(2*r, batch_size, false);
    matrix_multiply(a_cur, Wcur, Xin);

    Matrix h_prime(2*r, batch_size, false); 

    matrix_add(h_prime, a_prev, a_cur);
    matrix_bias(h_prime, h_prime, Bh);

    matrix_add(h_prime, h_prime, Lh);

    Matrix h(r, batch_size, false);

    for (int batch_idx=0; batch_idx<batch_size; batch_idx++) {
        for (int row = 0; row < r; row++) {
            h.set(row, batch_idx, tanh_proxy(h_prime.get(row, batch_idx)) * sigmoid_proxy(h_prime.get(row + r, batch_idx)));
        }
    }

    matrix_multiply(Xout, Wres, h);
    matrix_bias(Xout, Xout, Bres);
    matrix_add(Xout, Xout, Xin);

    matrix_multiply(skipOut, Wskip, h);
    matrix_add(skipOut, skipOut, skipIn);
    matrix_bias(skipOut,skipOut,Bskip);

    if (lastLayer) matrix_relu(skipOut, skipOut);

}

void nvWavenetFinal(Matrix& WskipOut, Matrix& BskipOut, Matrix& Wout, Matrix& Bout, Matrix& skip, Matrix& skipOut, Matrix& out, Matrix& p) {


    matrix_multiply(skipOut, WskipOut, skip);
    matrix_bias(skipOut, skipOut, BskipOut);
    matrix_relu(skipOut, skipOut);

    matrix_multiply(out, Wout, skipOut);
    matrix_bias(out, out, Bout);

    matrix_softmax(p, out);
}

void nvWavenetSelect(int sample, Matrix& p, Matrix& randomSelectors, std::vector<int>& y){

    for (int col=0; col<p.cols(); col++) {
        float sel = randomSelectors.get(col,sample);
        float sum = 0.f;
        y[col] = -1;
        for (int row=0; row<p.rows(); row++) {
            sum += p.get(row,col);
            if (sel < sum) {
                y[col] = row;
                break;
            }
        }
        assert(y[col] >= 0);
    }
}

nvWavenetReference::nvWavenetReference(int num_layers, int max_batch, int max_samples, int R, int S, int A, int max_dilation) : 
    m_numLayers(num_layers), m_maxBatch(max_batch), m_maxSamples(max_samples), m_R(R), m_S(S), m_A(A), m_maxDilation(max_dilation), m_lastSample(0) {
    m_embeddingsPrev = new Matrix(R,A);
    m_embeddingsCur = new Matrix(R,A);

    m_Wprev.resize(num_layers);
    m_Wcur.resize(num_layers);
    m_Bh.resize(num_layers);
    m_Wres.resize(num_layers);
    m_Bres.resize(num_layers);
    m_Wskip.resize(num_layers);
    m_Bskip.resize(num_layers);

    m_Xt.resize(max_samples);
    for (int sample = 0; sample < max_samples; sample++) {
        m_Xt[sample].resize(num_layers+1);
        for (int layer=0; layer<num_layers+1;layer++) {
            m_Xt[sample][layer] = new Matrix(R, max_batch);
        }
    }

    m_skipOut.resize(num_layers);


    for (int layer = 0; layer < num_layers; layer++) {
        m_Wprev[layer] = new Matrix(2*R,R);
        m_Wcur[layer] = new Matrix(2*R,R);
        m_Bh[layer] = new Matrix(2*R,1);
        m_Wres[layer] = new Matrix(R,R);
        m_Bres[layer] = new Matrix(R,1);
        m_Wskip[layer] = new Matrix(S,R);
        m_Bskip[layer] = new Matrix(S,1);

        m_skipOut[layer] = new Matrix(S, max_batch);
    }

    m_Lh.resize(max_samples);
    for (int sample = 0; sample < max_samples; sample++) {
        m_Lh[sample].resize(num_layers);
        for (int layer = 0; layer < num_layers; layer++) {
            m_Lh[sample][layer] = new Matrix(2*R, max_batch);
        }
    }

    m_Wzs = new Matrix(A,S);
    m_Bzs = new Matrix(A,1);
    m_Wza = new Matrix(A,A);
    m_Bza = new Matrix(A,1);

    m_yInPrev.resize(max_batch);
    m_yInCur.resize(max_batch);

    m_outputSelectors = new Matrix(max_batch,max_samples);

    m_Zs = new Matrix(A,max_batch);
    m_Za = new Matrix(A,max_batch);
    m_P  = new Matrix(A,max_batch);

}

nvWavenetReference::~nvWavenetReference() {
    delete m_embeddingsPrev;
    delete m_embeddingsCur;

    for (int sample=0; sample<m_maxSamples; sample++) {
        for (int layer=0; layer<m_numLayers; layer++) {
            delete m_Xt[sample][layer];
            delete m_Lh[sample][layer];
        }
    }

    for (int layer=0; layer<m_numLayers; layer++) {
        delete m_Wprev[layer];
        delete m_Wcur[layer];
        delete m_Bh[layer];
        delete m_Wres[layer];
        delete m_Bres[layer];
        delete m_Wskip[layer];

        delete m_skipOut[layer];
    }
    delete m_Wzs;
    delete m_Bzs;
    delete m_Wza;
    delete m_Bza;
    delete m_Zs;
    delete m_Za;
    delete m_P;
}

void nvWavenetReference::setEmbeddings (float* embedPrev, float* embedCur) {
    memcpy(m_embeddingsPrev->data(), embedPrev, m_R*m_A*sizeof(float));
    memcpy(m_embeddingsCur->data(), embedCur, m_R*m_A*sizeof(float));
}

void nvWavenetReference::setLayerWeights (int layer, float* Wprev, float* Wcur, float* Bh, float* Wres, float* Bres, float* Wskip, float* Bskip) {
    assert(layer<m_numLayers);
    memcpy(m_Wprev[layer]->data(), Wprev, 2*m_R*m_R*sizeof(float));
    memcpy(m_Wcur[layer]->data(), Wcur, 2*m_R*m_R*sizeof(float));
    memcpy(m_Bh[layer]->data(), Bh, 2*m_R*sizeof(float));
    memcpy(m_Wres[layer]->data(), Wres, m_R*m_R*sizeof(float));
    memcpy(m_Bres[layer]->data(), Bres, m_R*sizeof(float));
    memcpy(m_Wskip[layer]->data(), Wskip, m_S*m_R*sizeof(float));
    memcpy(m_Bskip[layer]->data(), Bskip, m_S*sizeof(float));
}

void nvWavenetReference::setOutWeights(float* Wzs, float* Bzs, float* Wza, float* Bza) {
    memcpy(m_Wzs->data(), Wzs, m_S*m_A*sizeof(float));
    memcpy(m_Bzs->data(), Bzs, m_A*sizeof(float));
    memcpy(m_Wza->data(), Wza, m_A*m_A*sizeof(float));
    memcpy(m_Bza->data(), Bza, m_A*sizeof(float));
}

void nvWavenetReference::setInputs(float* Lh, float* outputSelectors) {
    for (int i=0; i<m_maxBatch; i++) {
        m_yInPrev[i] = 128;
        m_yInCur[i] = 128;
    }
    for (int sample = 0; sample < m_maxSamples; sample++) {
        for (int layer = 0; layer < m_numLayers; layer++) {
            memcpy(m_Lh[sample][layer]->data(), Lh + sample*m_numLayers*m_maxBatch*2*m_R + layer*m_maxBatch*2*m_R, 2*m_R*m_maxBatch*sizeof(float));
        }
    }
    memcpy(m_outputSelectors->data(), outputSelectors, m_maxSamples*m_maxBatch*sizeof(int));
}

void nvWavenetReference::getXtOut(int layer, float* Xt) {
    memcpy(Xt, m_Xt[m_lastSample][layer+1]->data(), m_R*m_maxBatch*sizeof(float));
}

void nvWavenetReference::getSkipOut(int layer, float* hSkipOut) {
    memcpy(hSkipOut, m_skipOut[layer]->data(), m_S*m_maxBatch*sizeof(float));
}

void nvWavenetReference::getZs(float* hZs) {
    memcpy(hZs, m_Zs->data(), m_A*m_maxBatch*sizeof(float));
}

void nvWavenetReference::getZa(float* hZa) {
    memcpy(hZa, m_Za->data(), m_A*m_maxBatch*sizeof(float));
}

void nvWavenetReference::getP(float* hP) {
    memcpy(hP, m_P->data(), m_A*m_maxBatch*sizeof(float));
}

void nvWavenetReference::run(int num_samples, int batch_size, int* yOut) {
    Matrix zeroMatrixR(m_R,batch_size,false);
    for (int row=0; row<m_R; row++) {
        for (int col=0; col<batch_size; col++) {
            zeroMatrixR.set(row,col,0.f);
        }
    }
    Matrix zeroMatrixS(m_S,batch_size,false);
    for (int row=0; row<m_S; row++) {
        for (int col=0; col<batch_size; col++) {
            zeroMatrixS.set(row,col,0.f);
        }
    }

    for (int sample=0; sample<num_samples; sample++) {
        nvWavenetEmbed(m_yInPrev, m_yInCur, *m_embeddingsPrev, *m_embeddingsCur, *m_Xt[sample][0]);
        int dilation = 1;
        for (int l=0; l<m_numLayers; l++) {
            Matrix* Xtmd = (sample < dilation) ? &zeroMatrixR : m_Xt[sample-dilation][l];
            dilation *=2;
            if (dilation > m_maxDilation) dilation = 1;
            Matrix* skipIn = (l==0) ? &zeroMatrixS : m_skipOut[l-1];
            nvWavenetLayer(m_R, batch_size, *m_Wprev[l], *m_Wcur[l], *m_Bh[l], *m_Lh[sample][l], *m_Wres[l], *m_Bres[l], *m_Wskip[l], *m_Bskip[l], *Xtmd, *m_Xt[sample][l], *m_Xt[sample][l+1], *skipIn, *m_skipOut[l], l==m_numLayers-1);
        }
        nvWavenetFinal(*m_Wzs, *m_Bzs, *m_Wza, *m_Bza, *m_skipOut[m_numLayers-1], *m_Zs, *m_Za, *m_P);
        std::vector<int> yOut_sample(batch_size);
        nvWavenetSelect(sample, *m_P, *m_outputSelectors, yOut_sample);
        for (int b=0; b<batch_size; b++) {
            m_yInPrev[b] = m_yInCur[b];
            m_yInCur[b] = yOut_sample[b];
            yOut[b*num_samples + sample] = yOut_sample[b];
        }

    }
    m_lastSample = num_samples-1;
}
