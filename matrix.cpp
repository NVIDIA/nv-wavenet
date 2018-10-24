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
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

Matrix::Matrix(int rows, int cols, bool isTransposed) : m_rows(rows), m_cols(cols), m_isTransposed(isTransposed) {
    m_data = (float*)malloc(rows*cols*sizeof(float));
}

void Matrix::randomize(float mean, float scale, int sparsity) {
    for (int row = 0; row < rows(); row++) {
        for (int col = 0; col < cols(); col++) {

            if ((rand()%100) < sparsity) {
                set(row, col, 0.f);
            }
            else {
                // Generate a random number from 0 to 1.0
                float r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX); 
                // Convert to -.5 to .5
                r -= 0.5;
                // Scale and shift
                r = r * scale + mean;
                set(row, col, r);
            }
        }
    }
}

int Matrix::index(int row, int col) {
    if (m_isTransposed) {
        return col + row*m_cols; 
    }
    else {
        return row + col*m_rows;
    }
}

void Matrix::set(int row, int col, float val) { assert(row < m_rows); assert(col < m_cols); m_data[index(row,col)] = val; }

float Matrix::get(int row, int col) { assert(row < m_rows); assert(col < m_cols); return m_data[index(row,col)]; }

int Matrix::rows() { return m_rows; }

int Matrix::cols() { return m_cols; }

void Matrix::print(const char* name) {
    for (int row = 0; row < rows(); row++) {
        for (int col = 0; col < cols(); col++) {
            printf("%s[%d][%d] = %f\n", name, row, col, get(row,col));
        }
    }
}

float* Matrix::data() { return m_data; }

void matrix_multiply(Matrix& C, Matrix& A, Matrix& B)
{
    assert(A.rows() == C.rows());
    assert(B.cols() == C.cols());

    for (int row = 0; row != C.rows(); ++row) 
    {
        for (int col = 0; col != C.cols(); ++col)
        {
            float sum = 0;
            for (int inner = 0; inner != A.cols(); ++inner)
            {
                sum += A.get(row, inner) * B.get(inner, col);
            }
            C.set(row, col, sum);
        }
    }
}

void matrix_add(Matrix& C, Matrix& A, Matrix& B) {
    assert (A.rows() == B.rows());
    assert (A.rows() == C.rows());
    assert (A.cols() == B.cols());
    assert (A.cols() == C.cols());

    for (int row = 0; row != C.rows(); ++row) {
        for (int col = 0; col < C.cols(); ++col) {
            C.set(row, col, A.get(row, col) + B.get(row, col));
        }
    }
}

void matrix_bias(Matrix& C, Matrix& A, Matrix& B) {
    assert(A.rows() == C.rows());
    assert(A.rows() == B.rows());
    assert(A.cols() == C.cols());
    assert(B.cols() == 1);

    for (int row = 0; row != C.rows(); ++row) {
        for (int col = 0; col < C.cols(); ++col) {
            C.set(row, col, A.get(row, col) + B.get(row, 0));
        }
    }

}

void matrix_compare(const char* name, Matrix& A, Matrix& B, float max_error, bool relu) {
    assert(A.rows() == B.rows());
    assert(A.cols() == B.cols());

    printf("Comparing %s\n", name);
    for (int row =0; row < A.rows(); row++) {
        for (int col=0; col < A.cols(); col++) {
            float A_data = A.get(row,col);
            float B_data = B.get(row,col);

            bool correct = false;
            if (relu && (A_data <= 0.f || B_data <= 0.f)) correct = A_data < max_error && B_data < max_error;
            else correct = (fabs(B_data/A_data)-1) <= max_error;
            if (!correct) {
                printf("  mismatch at %d,%d: %.10e vs %.10e\n", row, col, A_data, B_data);
                assert(false);
            }
        }
    }
    printf("  SUCCESS!\n");

}

void matrix_relu(Matrix& dst, Matrix& src) {
    assert(src.rows() == dst.rows());
    assert(src.cols() == dst.cols());

    for (int row=0; row < src.rows(); row++) {
        for (int col=0; col < src.cols(); col++) {
            float srcVal = src.get(row,col);
            float dstVal = (srcVal < 0) ? 0.f : srcVal;
            dst.set(row,col,dstVal);
        }
    }
}

void matrix_softmax(Matrix& dst, Matrix& src) {
    assert(dst.rows() == src.rows());
    assert(dst.cols() == src.cols());
    for (int col = 0; col < src.cols(); col++) {
        float max = 0.f;
        for (int row=0; row<src.rows();row++) {
            if (src.get(row,col) > max) max = src.get(row,col);
        }
        float sum = 0.f;
        for (int row = 0; row < src.rows(); row++) {
            sum += exp(src.get(row,col) - max);
        }
        for (int row = 0; row < src.rows(); row++) {
            dst.set(row, col, exp(src.get(row,col)-max)/sum);
        }
    }
}
