#include "Cell.h"

__kernel void square(
	__global Cell* input,
	__global Cell* output,
	int sizeX,
	int sizeY) 
{
	int i = get_global_id(0);
	int j = get_global_id(1);
	if(i < sizeX && j < sizeY) {
		int e = i + sizeX * j;
		output[e].value = input[e].value * input[e].value;
	}
}

