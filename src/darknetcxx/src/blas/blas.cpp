#include "darknetcxx/blas.hpp"
#include <algorithm>
#include <cassert>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// no raw loops

namespace darknet {
void
reorg_cpu(float* x, int w, int h, int c, int batch, int stride, int forward, float* out)
{
    int b, i, j, k;
    int out_c = c / (stride * stride);

    for (b = 0; b < batch; ++b) {
        for (k = 0; k < c; ++k) {
            for (j = 0; j < h; ++j) {
                for (i = 0; i < w; ++i) {
                    int in_index = i + w * (j + h * (k + c * b));
                    int c2 = k % out_c;
                    int offset = k / out_c;
                    int w2 = i * stride + offset % stride;
                    int h2 = j * stride + offset / stride;
                    int out_index = w2 + w * stride * (h2 + h * stride * (c2 + out_c * b));
                    if (forward)
                        out[out_index] = x[in_index];
                    else
                        out[in_index] = x[out_index];
                }
            }
        }
    }
}

void
flatten(float* x, int size, int layers, int batch, int forward)
{
    // float *swap = calloc(size * layers * batch, sizeof(float));
    float* swap = new float[size * layers * batch];
    int i, c, b;
    for (b = 0; b < batch; ++b) {
        for (c = 0; c < layers; ++c) {
            for (i = 0; i < size; ++i) {
                int i1 = b * layers * size + c * size + i;
                int i2 = b * layers * size + i * layers + c;
                if (forward)
                    swap[i2] = x[i1];
                else
                    swap[i1] = x[i2];
            }
        }
    }
    memcpy(x, swap, size * layers * batch * sizeof(float));
    // free(swap);
    delete[] swap; // exception unsafe
}

void
weighted_sum_cpu(float* a, float* b, float* s, int n, float* c)
{
    int i;
    for (i = 0; i < n; ++i) {
        c[i] = s[i] * a[i] + (1 - s[i]) * (b ? b[i] : 0);
    }
}

void
shortcut_cpu(int batch, int w1, int h1, int c1, float* add, int w2, int h2, int c2, float* out)
{
    int stride = w1 / w2;
    int sample = w2 / w1;
    assert(stride == h1 / h2);
    assert(sample == h2 / h1);
    if (stride < 1)
        stride = 1;
    if (sample < 1)
        sample = 1;
    int minw = (w1 < w2) ? w1 : w2;
    int minh = (h1 < h2) ? h1 : h2;
    int minc = (c1 < c2) ? c1 : c2;

    int i, j, k, b;
    for (b = 0; b < batch; ++b) {
        for (k = 0; k < minc; ++k) {
            for (j = 0; j < minh; ++j) {
                for (i = 0; i < minw; ++i) {
                    int out_index = i * sample + w2 * (j * sample + h2 * (k + c2 * b));
                    int add_index = i * stride + w1 * (j * stride + h1 * (k + c1 * b));
                    out[out_index] += add[add_index];
                }
            }
        }
    }
}

void
mean_cpu(float* x, int batch, int depth, int spatial, float* mean)
{
    float scale = 1.f / (batch * spatial);
    for (int c = 0; c < depth; ++c) {
        mean[c] = 0;
        for (int b = 0; b < batch; ++b) {
            for (int k = 0; k < spatial; ++k) {
                int index = b * depth * spatial + c * spatial + k;
                mean[c] += x[index];
            }
        }
        mean[c] *= scale;
    }
}

void
variance_cpu(float* x, float* mean, int batch, int depth, int spatial, float* variance)
{
    // TODO FIXME is that reasonable, why - 1?
    float scale = 1. / (batch * spatial - 1);
    for (int i = 0; i < depth; ++i) {
        variance[i] = 0;
        for (int j = 0; j < batch; ++j) {
            for (int k = 0; k < spatial; ++k) {
                int index = j * depth * spatial + i * spatial + k;
                variance[i] += pow((x[index] - mean[i]), 2);
            }
        }
        variance[i] *= scale;
    }
}

void
normalize_cpu(float* x, float* mean, float* variance, int batch, int depth, int spatial)
{
    int b, f, i;
    for (b = 0; b < batch; ++b) {
        for (f = 0; f < depth; ++f) {
            for (i = 0; i < spatial; ++i) {
                int index = b * depth * spatial + f * spatial + i;
                x[index] = (x[index] - mean[f]) / (sqrt(variance[f]) + .000001f);
            }
        }
    }
}

void
const_cpu(int N, float ALPHA, float* X, int INCX)
{
    int i;
    for (i = 0; i < N; ++i)
        X[i * INCX] = ALPHA;
}

void
mul_cpu(int N, float* X, int INCX, float* Y, int INCY)
{
    int i;
    for (i = 0; i < N; ++i)
        Y[i * INCY] *= X[i * INCX];
}

// TODO: inline and for_each every function here ...
void
pow_cpu(int N, float ALPHA, float* X, int INCX, float* Y, int INCY)
{
    int i;
    for (i = 0; i < N; ++i)
        Y[i * INCY] = pow(X[i * INCX], ALPHA);
}

void
axpy_cpu(int N, float ALPHA, float* X, float* Y)
{
    for (int i = 0; i < N; ++i)
        Y[i] += ALPHA * X[i];
}

void
scal_cpu(int N, float ALPHA, float* X)
{
    for (int i = 0; i < N; ++i)
        X[i] *= ALPHA;
}

// TODO: inline ?
void
fill_cpu(int N, float ALPHA, float* X)
{
    std::fill_n(X, N, ALPHA);
    // int i;
    // for (i = 0; i < N; ++i)
    //   X[i * INCX] = ALPHA;
}

void
copy_cpu(int N, float* X, float* Y)
{
    std::copy_n(X, N, Y);
    // int i;
    // for (i = 0; i < N; ++i)
    //   Y[i * INCY] = X[i * INCX];
}

void
smooth_l1_cpu(int n, float* pred, float* truth, float* delta, float* error)
{
    int i;
    for (i = 0; i < n; ++i) {
        float diff = truth[i] - pred[i];
        float abs_val = fabs(diff);
        if (abs_val < 1) {
            error[i] = diff * diff;
            delta[i] = diff;
        } else {
            error[i] = 2 * abs_val - 1;
            delta[i] = (diff < 0) ? -1 : 1;
        }
    }
}

void
l2_cpu(int n, float* pred, float* truth, float* delta, float* error)
{
    int i;
    for (i = 0; i < n; ++i) {
        float diff = truth[i] - pred[i];
        error[i] = diff * diff;
        delta[i] = diff;
    }
}

float
dot_cpu(int N, float* X, int INCX, float* Y, int INCY)
{
    int i;
    float dot = 0;
    for (i = 0; i < N; ++i)
        dot += X[i * INCX] * Y[i * INCY];
    return dot;
}

void
softmax(float* input, int n, float temp, float* output)
{
    int i;
    float sum = 0;
    float largest = -FLT_MAX;
    for (i = 0; i < n; ++i) {
        if (input[i] > largest)
            largest = input[i];
    }
    for (i = 0; i < n; ++i) {
        float e = exp(input[i] / temp - largest / temp);
        sum += e;
        output[i] = e;
    }
    for (i = 0; i < n; ++i) {
        output[i] /= sum;
    }
}

void
add_bias(float* output, float* biases, int batch, int n, int size)
{
    int i, j, b;
    for (b = 0; b < batch; ++b) {
        for (i = 0; i < n; ++i) {
            for (j = 0; j < size; ++j) {
                output[(b * n + i) * size + j] += biases[i];
            }
        }
    }
}

void
scale_bias(float* output, float* scales, int batch, int n, int size)
{
    int i, j, b;
    for (b = 0; b < batch; ++b) {
        for (i = 0; i < n; ++i) {
            for (j = 0; j < size; ++j) {
                output[(b * n + i) * size + j] *= scales[i];
            }
        }
    }
}

void
backward_scale_cpu(float* x_norm, float* delta, int batch, int n, int size, float* scale_updates)
{
    int i, b, f;
    for (f = 0; f < n; ++f) {
        float sum = 0;
        for (b = 0; b < batch; ++b) {
            for (i = 0; i < size; ++i) {
                int index = i + size * (f + n * b);
                sum += delta[index] * x_norm[index];
            }
        }
        scale_updates[f] += sum;
    }
}

void
mean_delta_cpu(float* delta, float* variance, int batch, int filters, int spatial, float* mean_delta)
{
    int i, j, k;
    for (i = 0; i < filters; ++i) {
        mean_delta[i] = 0;
        for (j = 0; j < batch; ++j) {
            for (k = 0; k < spatial; ++k) {
                int index = j * filters * spatial + i * spatial + k;
                mean_delta[i] += delta[index];
            }
        }
        mean_delta[i] *= (-1. / sqrt(variance[i] + .00001f));
    }
}
void
variance_delta_cpu(float* x, float* delta, float* mean, float* variance, int batch, int filters, int spatial, float* variance_delta)
{
    int i, j, k;
    for (i = 0; i < filters; ++i) {
        variance_delta[i] = 0;
        for (j = 0; j < batch; ++j) {
            for (k = 0; k < spatial; ++k) {
                int index = j * filters * spatial + i * spatial + k;
                variance_delta[i] += delta[index] * (x[index] - mean[i]);
            }
        }
        variance_delta[i] *= -.5 * pow(variance[i] + .00001f, (float)(-3. / 2.));
    }
}
void
normalize_delta_cpu(float* x, float* mean, float* variance, float* mean_delta, float* variance_delta, int batch, int filters, int spatial, float* delta)
{
    int f, j, k;
    for (j = 0; j < batch; ++j) {
        for (f = 0; f < filters; ++f) {
            for (k = 0; k < spatial; ++k) {
                int index = j * filters * spatial + f * spatial + k;
                delta[index] = delta[index] * 1. / (sqrt(variance[f]) + .00001f) + variance_delta[f] * 2. * (x[index] - mean[f]) / (spatial * batch) + mean_delta[f] / (spatial * batch);
            }
        }
    }
}

} // namespace darknet
