#include "local.h"

void arm_convolve_HWC_q15_nonsquare_ref(const q15_t * Im_in,
                          const uint16_t dim_im_in_x,
                          const uint16_t dim_im_in_y,
                          const uint16_t ch_im_in,
                          const q15_t * wt,
                          const uint16_t ch_im_out,
                          const uint16_t dim_kernel_x,
                          const uint16_t dim_kernel_y,
                          const uint16_t padding_x,
                          const uint16_t padding_y,
                          const uint16_t stride_x,
                          const uint16_t stride_y,
                          const q15_t * bias,
                          const uint16_t bias_shift,
                          const uint16_t out_shift,
                          q15_t * Im_out,
                          const uint16_t dim_im_out_x,
                          const uint16_t dim_im_out_y, 
                          q15_t * bufferA, 
                          q7_t * bufferB)

{	
    uint16_t  i, j, k, l, m, n;
    int       conv_out;
    uint16_t in_row, in_col;

    for (i = 0; i < ch_im_out; i++)
    {
        for (j = 0; j < dim_im_out_y; j++)
        {
            for (k = 0; k < dim_im_out_x; k++)
            {
#ifndef ARM_NN_TRUNCATE
                conv_out = (bias[i] << bias_shift) + (0x1 << (out_shift - 1));
#else
                conv_out = bias[i] << bias_shift;
#endif
                for (m = 0; m < dim_kernel_y; m++)
                {
                    for (n = 0; n < dim_kernel_x; n++)
                    {
                        in_row = stride_y * j + m - padding_y;
                        in_col = stride_x * k + n - padding_x;
                        if (in_row >= 0 && in_col >= 0 && in_row < dim_im_in_y && in_col < dim_im_in_x)
                        {
                            for (l = 0; l < ch_im_in; l++)
                            {
                                conv_out +=
                                    Im_in[(in_row * dim_im_in_x + in_col) * ch_im_in +
                                          l] * wt[i * ch_im_in * dim_kernel_x * dim_kernel_y + (m * dim_kernel_x +
                                                                                            n) * ch_im_in + l];
                            }
                        }
                    }
                }
                Im_out[i + (j * dim_im_out_x + k) * ch_im_out] = (q15_t) __SSAT((conv_out >> out_shift), 16);
            }
        }
    }
}	


void local_maxpool_q15_HWC(const q15_t *Im_in,           // input image
	const uint16_t dim_im_in_x,  // input image dimension x or W
	const uint16_t dim_im_in_y,  // input image dimension y or H
	const uint16_t ch_im_in,     // number of input image channels
	const uint16_t dim_kernel_x, // window kernel size
	const uint16_t dim_kernel_y, // window kernel size
	const uint16_t padding_x,    // padding sizes
	const uint16_t padding_y,    // padding sizes
	const uint16_t stride_x,     // stride
	const uint16_t stride_y,     // stride
	const uint16_t dim_im_out_x, // output image dimension x or W
	const uint16_t dim_im_out_y, // output image dimension y or H
	q7_t *bufferA,               // a buffer for local storage, NULL by now
	q15_t *Im_out)
{
    int16_t i_ch_in, i_x, i_y;
    int16_t k_x, k_y;

    for (i_ch_in = 0; i_ch_in < ch_im_in; i_ch_in++)
    {
        for (i_y = 0; i_y < dim_im_out_y; i_y++)
        {
            for (i_x = 0; i_x < dim_im_out_x; i_x++)
            {
                int max = -32768;
                for (k_y = i_y * stride_y - padding_y; k_y < i_y * stride_y - padding_y + dim_kernel_y; k_y++)
                {
                    for (k_x = i_x * stride_x - padding_x; k_x < i_x * stride_x - padding_x + dim_kernel_x; k_x++)
                    {
                        if (k_y >= 0 && k_x >= 0 && k_y < dim_im_in_y && k_x < dim_im_in_x)
                        {
                            if (Im_in[i_ch_in + ch_im_in * (k_x + k_y * dim_im_in_x)] > max)
                            {
                                max = Im_in[i_ch_in + ch_im_in * (k_x + k_y * dim_im_in_x)];
                            }
                        }
                    }
                }
                Im_out[i_ch_in + ch_im_in * (i_x + i_y * dim_im_out_x)] = max;
            }
        }
    }
}



void global_avepool_q15_HWC(const q15_t *Im_in,           // input image
	const uint16_t dim_im_in_x,  // input image dimension x or W
	const uint16_t dim_im_in_y,  // input image dimension y or H
	const uint16_t ch_im_in,     // number of input image channels
	q15_t *Im_out)
{
	uint16_t i;
	uint16_t j;
	int size = dim_im_in_x * dim_im_in_y;
	int size_t = size * ch_im_in;

	for (j = 0; j < ch_im_in; j++) {
		int sum = 0;
		for (i = j; i < size_t; i += ch_im_in) {
			sum += Im_in[i];
		}
		Im_out[j] = sum / size;
	}
}