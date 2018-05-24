#pragma once

void im2col_cpu(float *data_im, int channels, int height, int width, int ksize,
                int stride, int pad, float *data_col);

void im2col_ongpu(float *im, int channels, int height, int width, int ksize,
                  int stride, int pad, float *data_col);

void col2im_cpu(float *data_col, int channels, int height, int width, int ksize,
                int stride, int pad, float *data_im);

void col2im_ongpu(float *data_col, int channels, int height, int width,
                  int ksize, int stride, int pad, float *data_im);
