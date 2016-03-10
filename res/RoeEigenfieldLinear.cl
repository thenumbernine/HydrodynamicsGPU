#include "HydroGPU/Roe.h"

/*
default implementation assumes eigenvector are inverse eigenvector matrices
and EIGEN_TRANSFORM_STRUCT_SIZE == 2 * NUM_STATES
*/
#if EIGEN_TRANSFORM_STRUCT_SIZE != 2 * NUM_STATES * NUM_STATES
#error expected eignvectors to be square matrices size of NUM_STATES
#endif
#if EIGEN_SPACE_DIM != NUM_STATES
#error expected eigen space dim to match number of states
#endif

//c_i = a_ij b_j

void stateMatrixTransform_G_(
	real* results,
	const __global real* matrix,
	const real* input);

void stateMatrixTransform_G_(
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

// eigenvector functions

void leftEigenvectorTransform(
	real* results,
	const __global real* eigenvector,
	const real* input,
	int side)
{
	stateMatrixTransform_G_(results, eigenvector, input);
}

void rightEigenvectorTransform(
	__global real* results,
	const __global real* eigenvector,
	const real* input,
	int side)
{
	stateMatrixTransformGG_(results, eigenvector + NUM_STATES * NUM_STATES, input);
}
