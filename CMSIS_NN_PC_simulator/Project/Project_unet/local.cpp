#include "local.h"

int ssat(int a, const int bit) {
	int clip_value = a;
	if (a > (0x1 << (bit - 1))-1) {
		clip_value = (0x1 << (bit - 1)) - 1;
	}
	else if (a < -(0x1 << (bit - 1))) {
		clip_value = -(0x1 << (bit - 1));
	}
	return clip_value;
};


void arm_convolve_HWC_q15_fast_nonsquare(const q15_t * Im_in,
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

				//conv_out = (bias[i] << bias_shift) + (0x1 << (out_shift - 1));
				conv_out = bias[i] << bias_shift;
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
				Im_out[i + (j * dim_im_out_x + k) * ch_im_out] = (q15_t)ssat((conv_out >> out_shift), 16);
			}
		}
	}
}


void arm_fully_connected_q15_ref(const q15_t * pV,  // pointer to vector
	const q15_t * pM,  // pointer to matrix
	const uint16_t dim_vec,    // length of the vector
	const uint16_t num_of_rows,    // numCol of A
	const uint16_t bias_shift, // amount of left-shift for bias
	const uint16_t out_shift,  // amount of right-shift for output
	const q15_t * bias, q15_t * pOut,  // output operand
	q15_t * vec_buffer)
{
	for (int i = 0; i < num_of_rows; i++)
	{

		int       ip_out = bias[i] << bias_shift;
		for (int j = 0; j < dim_vec; j++)
		{
			ip_out += pV[j] * pM[i * dim_vec + j];
		}
		pOut[i] = (q15_t)ssat((ip_out >> out_shift), 16);
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

// modified from CMSIS-NN test_ref
void local_avepool_q15_HWC(const q15_t *Im_in,           // input image
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
				int sum = 0;
				int count = 0;
				for (k_y = i_y * stride_y - padding_y; k_y < i_y * stride_y - padding_y + dim_kernel_y; k_y++)
				{
					for (k_x = i_x * stride_x - padding_x; k_x < i_x * stride_x - padding_x + dim_kernel_x; k_x++)
					{
						if (k_y >= 0 && k_x >= 0 && k_y < dim_im_in_y && k_x < dim_im_in_x)
						{
							sum += Im_in[i_ch_in + ch_im_in * (k_x + k_y * dim_im_in_x)];
							count++;
						}
					}
				}
				Im_out[i_ch_in + ch_im_in * (i_x + i_y * dim_im_out_x)] = sum / count;
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


void local_up_sampling_q15_HWC(const q15_t *Im_in,       // input image
	const uint16_t dim_im_in_x,  // input image dimension x or W
	const uint16_t dim_im_in_y,  // input image dimension y or H
	const uint16_t ch_im_in,     // number of input image channels
	const uint16_t dim_kernel_x, // window kernel size
	const uint16_t dim_kernel_y, // window kernel size
	const uint16_t dim_im_out_x, // output image dimension x or W
	const uint16_t dim_im_out_y, // output image dimension y or H
	q7_t *bufferA,               // a buffer for local storage, NULL by now
	q15_t *Im_out)
{
	int16_t i_x, i_y;

	// for loop for each pixel in input image.
	for (i_y = 0; i_y < dim_im_in_y; i_y++)
	{
		for (i_x = 0; i_x < dim_im_in_x; i_x++)
		{
			// copy all the channels together. 
			const q15_t *p_in = Im_in + (i_y * dim_im_in_x + i_x) * ch_im_in;
			q15_t *pout = Im_out + (i_y * dim_im_in_x * dim_kernel_x * dim_kernel_y + i_x * dim_kernel_x) * ch_im_in;

			// copy along x axis
			for (int i = 0; i<dim_kernel_x; i++)
				memcpy(pout + i * ch_im_in, p_in, ch_im_in * sizeof(q15_t));
			// duplicate the copied x data into y axis. 
			for (int i = 1; i<dim_kernel_y; i++)
				memcpy(pout + i * ch_im_in * dim_im_in_x * dim_kernel_x, pout, ch_im_in * dim_kernel_x * sizeof(q15_t));
		}
	}
}


void arm_relu_q15(q15_t * data, uint16_t size)
{
	uint16_t  i;

	for (i = 0; i < size; i++)
	{
		if (data[i] < 0)
			data[i] = 0;
	}
}


void local_cat_q15_HWC(
	const q15_t *Im_in_1,
	const q15_t *Im_in_2,
	const uint16_t ch_im_in_1,
	const uint16_t ch_im_in_2,
	const uint16_t dim_im_out_x,
	const uint16_t dim_im_out_y,
	q15_t *Im_out)
{
	uint16_t i;
	uint16_t size = dim_im_out_x * dim_im_out_y;
	for (i = 0; i<size; i++) {
		memcpy(Im_out + i*(ch_im_in_1 + ch_im_in_2), Im_in_1 + i*ch_im_in_1, ch_im_in_1 * sizeof(q15_t));
		memcpy(Im_out + i*(ch_im_in_1 + ch_im_in_2) + ch_im_in_1, Im_in_2 + i*ch_im_in_2, ch_im_in_2 * sizeof(q15_t));
	}
}


void local_tsm_q15_HWC(
	q15_t *Im_in,
	const uint16_t ch_im_in,
	const uint16_t dim_im_in_x,
	const uint16_t dim_im_in_y) 
{
	uint16_t shift_ch = (uint16_t)(ch_im_in / 4);
	uint16_t i, j;
	for (i = 0; i < dim_im_in_y - 1; i++) {
		for (j = 0; j < dim_im_in_x; j++) {
			memcpy(Im_in + dim_im_in_x*ch_im_in*i + j*ch_im_in, Im_in + dim_im_in_x*ch_im_in*(i+1) + j*ch_im_in, shift_ch * sizeof(q15_t));
		}
	}

	for (i = dim_im_in_y - 1; i > 0; i--) {
		for (j = 0; j < dim_im_in_x; j++) {
			memcpy(Im_in + dim_im_in_x*ch_im_in*i + j*ch_im_in + shift_ch, Im_in + dim_im_in_x*ch_im_in*(i-1) + j*ch_im_in + shift_ch, shift_ch * sizeof(q15_t));
		}
	}
}