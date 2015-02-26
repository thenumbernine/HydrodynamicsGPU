#include "HydroGPU/Shared/Common.h"

/*
default implementation assumes eigenfield are inverse eigenvector matrices
and EIGENFIELD_SIZE == NUM_STATES
*/
#if EIGENFIELD_SIZE != NUM_STATES * NUM_STATES
#error expected eignfields to be square matrices size of NUM_STATES
#endif

//c_i = a_ij b_j
void stateMatrixTransform(
	real* results,
	const __global real* matrix,
	const real* input);

void stateMatrixTransform(
	real* results,
	const __global real* matrix,
	const real* input)
{
	for (int i = 0; i < NUM_STATES; ++i) {
		real sum = 0.f;
		for (int j = 0; j < NUM_STATES; ++j) {
			sum += matrix[i + NUM_STATES * j] * input[j];
		}
		results[i] = sum;
	}
}

void eigenfieldTransform(
	real* results,
	const __global real* eigenfield,
	const real* input);

void eigenfieldTransform(
	real* results,
	const __global real* eigenfield,
	const real* input)
{
	stateMatrixTransform(results, eigenfield, input);
}

void eigenfieldInverseTransform(
	real* results,
	const __global real* eigenfieldInverse,
	const real* input);

void eigenfieldInverseTransform(
	real* results,
	const __global real* eigenfieldInverse,
	const real* input)
{
	stateMatrixTransform(results, eigenfieldInverse, input);
}

