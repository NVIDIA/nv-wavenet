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

#ifndef __MATRIX__
#define __MATRIX__

class Matrix {
    private:
        float* m_data;
        bool   m_isTransposed;
        int    m_rows;
        int    m_cols;

    public:
        Matrix(int rows, int cols, bool isTransposed=false);

        void randomize(float mean, float scale, int sparsity = 0);
        
        int index(int row, int col); 

        void set(int row, int col, float val);

        float get(int row, int col);

        int rows();

        int cols();

        void print(const char* name); 

        float* data();
};

void matrix_multiply(Matrix& C, Matrix& A, Matrix& B);
void matrix_add(Matrix& C, Matrix& A, Matrix& B);
void matrix_bias(Matrix& C, Matrix&A, Matrix& B);
void matrix_compare(const char* name, Matrix& A, Matrix& B, float max_error=1.e-6, bool relu=false);
void matrix_relu(Matrix& dst, Matrix& src); 
void matrix_softmax(Matrix& dst, Matrix& src);

#endif
