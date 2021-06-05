#ifndef __NN_H__
#define __NN_H__

#include "mbed.h"
#include "arm_math.h"
#include "parameter.h"
#include "weight.h"
#include "arm_nnfunctions.h"
#include "local.h"
#include "input_output.h"

void run_nn(q15_t* input_data, q15_t* output_data);

#endif