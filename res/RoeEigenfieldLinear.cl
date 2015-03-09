#include "HydroGPU/Roe.h"

/*
default implementation assumes eigenfield are inverse eigenvector matrices
and EIGEN_TRANSFORM_STRUCT_SIZE == 2 * NUM_STATES
*/
#if EIGEN_TRANSFORM_STRUCT_SIZE != 2 * NUM_STATES * NUM_STATES
#error expected eignfields to be square matrices size of NUM_STATES
#endif

//c_i = a_ij b_j

void stateMatrixTransform_GG(
	real* results,
	const __global real* matrix,
	const __global real* input);

void stateMatrixTransform_GG(
	real* results,
	const __global real* matrix,
	const __global real* input)
{
	for (int i = 0; i < NUM_STATES; ++i) {
		real sum = 0.f;
		for (int j = 0; j < NUM_STATES; ++j) {
			sum += matrix[i + NUM_STATES * j] * input[j];
		}
		results[i] = sum;
	}
}

// same as above but with global, global, local parameters

void stateMatrixTransformGG_(
	__global real* results,
	const __global real* matrix,
	const real* input);

void stateMatrixTransformGG_(
	__global real* results,
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

// eigenfield functions

void eigenfieldTransform(
	real* results,
	const __global real* eigenfield,
	const __global real* input,
	int side)
{
	stateMatrixTransform_GG(results, eigenfield, input);
}

void eigenfieldInverseTransform(
	__global real* results,
	const __global real* eigenfield,
	const real* input,
	int side)
{
	stateMatrixTransformGG_(results, eigenfield + NUM_STATES * NUM_STATES, input);
}

