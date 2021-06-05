#include "nn.h"
#include "sample_input_output.h"
#include <iostream>
using namespace std;

void main() {
	q15_t test_input[4096] = INPUT_DATA;
	//q15_t test_input[96] = POOL;
	q15_t test_output[1] = FC3;
	q15_t output_buffer[1];

	time_t begin_t = clock();
	run_nn(test_input, output_buffer);
	time_t finish_t = clock();
	cout << "total_time: " << (double)(finish_t - begin_t) / CLOCKS_PER_SEC << "s" << endl;
	
	for (int i = 0; i < 1; i++) {
		cout << test_output[i] << " " << output_buffer[i] << endl;
	}

	system("pause");
}