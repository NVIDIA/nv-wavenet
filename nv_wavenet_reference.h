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
#include <vector>

void nvWavenetEmbed(std::vector<int>& yInPrev, std::vector<int>& yInCur, Matrix& embeddingsPrev, Matrix& embeddingsCur, Matrix& x);
void nvWavenetLayer(int r, int batch_size, Matrix& Wprev, Matrix& Wcur, Matrix& Bh, Matrix& Lh, Matrix& Wres, Matrix& Bres, Matrix& Wskip, Matrix& Bskip, Matrix& Xtmd, Matrix& Xin, Matrix& Xout, Matrix& skipIn, Matrix& skipOut, bool lastLayer); 
void nvWavenetFinal(Matrix& WskipOut, Matrix& BskipOut, Matrix& Wout, Matrix& Bout, Matrix& skip, Matrix& skipOut, Matrix& out, Matrix& p);
void nvWavenetSelect(int sample, Matrix& p, Matrix& randomSelectors, std::vector<int>& y);

class nvWavenetReference {
    private:

        int m_numLayers;
        int m_maxBatch;
        int m_maxSamples;

        int m_R;
        int m_S;
        int m_A;

        int m_maxDilation;

        Matrix* m_embeddingsPrev;
        Matrix* m_embeddingsCur;

        std::vector<Matrix*> m_Wprev;
        std::vector<Matrix*> m_Wcur;
        std::vector<Matrix*> m_Bh;
        std::vector< std::vector<Matrix*> > m_Lh;
        std::vector<Matrix*> m_Wres;
        std::vector<Matrix*> m_Bres;
        std::vector<Matrix*> m_Wskip;
        std::vector<Matrix*> m_Bskip;

        Matrix* m_Wzs;
        Matrix* m_Bzs;
        Matrix* m_Wza;
        Matrix* m_Bza;

        std::vector<int> m_yInPrev;
        std::vector<int> m_yInCur;

        Matrix* m_outputSelectors;

        int m_lastSample;

        std::vector< std::vector<Matrix*> > m_Xt;
        std::vector<Matrix*> m_skipOut;

        Matrix* m_Zs;
        Matrix* m_Za;
        Matrix* m_P;


    public:

        nvWavenetReference(int num_layers, int max_batch, int max_samples, int R, int S, int A, int max_dilation);
        ~nvWavenetReference();

        // Model initialization
        void setEmbeddings (float* embedPrev, float* embedCur); 
        void setLayerWeights (int layer, float* Wprev, float* Wcur, float* Bh, float* Wres, float* Bres, float* Wskip, float* Bskip); 
        void setOutWeights (float* Wzs, float* Bzs, float* Wza, float* Bza); 
        void setInputs (float* Lh, float* outputSelectors); 

        // Fetch intermediate results
        void getXtOut(int layer, float* Xt);
        void getSkipOut(int layer, float* SkipOut);
        void getZs(float* Zs);
        void getZa(float* Za);
        void getP(float* P);

        // Run
        void run(int num_samples, int batch_size, int* yOut);
};
