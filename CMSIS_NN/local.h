#include "arm_math.h"

// https://github.com/STMicroelectronics/STM32CubeF1/blob/master/Drivers/CMSIS/NN/NN_Lib_Tests/nn_test/Ref_Implementations/arm_convolve_HWC_q15_ref_nonsquare.c
void arm_convolve_HWC_q15_nonsquare_ref(
	const q15_t * Im_in,
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
  q7_t * bufferB);

// https://github.com/majianjia/nnom/blob/master/src/backends/nnom_local_q15.c
void local_maxpool_q15_HWC(
	const q15_t *Im_in,           // input image
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
	q15_t *Im_out);

void global_avepool_q15_HWC(const q15_t *Im_in,           // input image
	const uint16_t dim_im_in_x,  // input image dimension x or W
	const uint16_t dim_im_in_y,  // input image dimension y or H
	const uint16_t ch_im_in,     // number of input image channels
	q15_t *Im_out);
	