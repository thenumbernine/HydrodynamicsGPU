#include "HydroGPU/Shared/Common.h"

constant int4 stepsize = (int4)(STEP_X, STEP_Y, STEP_Z, STEP_W);
constant real4 dx = (real4)(DX, DY, DZ, 1.f);

//http://developer.amd.com/resources/documentation-articles/articles-whitepapers/opencl-optimization-case-study-simple-reductions/
__kernel void calcCFLMinReduce(
	const __global real* buffer,
	__local real* scratch,
	__const int length,
	__global real* result)
{
	int global_index = get_global_id(0);
	real accumulator = INFINITY;
	// Loop sequentially over chunks of input vector
	while (global_index < length) {
		real element = buffer[global_index];
		accumulator = (accumulator < element) ? accumulator : element;
		global_index += get_global_size(0);
	}

	// Perform parallel reduction
	int local_index = get_local_id(0);
	scratch[local_index] = accumulator;
	barrier(CLK_LOCAL_MEM_FENCE);
	for(int offset = get_local_size(0) / 2; offset > 0; offset = offset / 2) {
		if (local_index < offset) {
			real other = scratch[local_index + offset];
			real mine = scratch[local_index];
			scratch[local_index] = (mine < other) ? mine : other;
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	if (local_index == 0) {
		result[get_group_id(0)] = scratch[0];
	}
}

//periodic

__kernel void stateBoundaryPeriodicX(
	__global real* stateBuffer)
{
	int2 i = (int2)(get_global_id(0), get_global_id(1));
	for (int j = 0; j < NUM_STATES; ++j) {
		stateBuffer[j + NUM_STATES * INDEX(0, i.x, i.y)] = stateBuffer[j + NUM_STATES * INDEX(SIZE_X - 4, i.x, i.y)];
		stateBuffer[j + NUM_STATES * INDEX(1, i.x, i.y)] = stateBuffer[j + NUM_STATES * INDEX(SIZE_X - 3, i.x, i.y)];
		stateBuffer[j + NUM_STATES * INDEX(SIZE_X - 2, i.x, i.y)] = stateBuffer[j + NUM_STATES * INDEX(2, i.x, i.y)];
		stateBuffer[j + NUM_STATES * INDEX(SIZE_X - 1, i.x, i.y)] = stateBuffer[j + NUM_STATES * INDEX(3, i.x, i.y)];
	}
}

__kernel void stateBoundaryPeriodicY(
	__global real* stateBuffer)
{
	int2 i = (int2)(get_global_id(0), get_global_id(1));
	for (int j = 0; j < NUM_STATES; ++j) {
		stateBuffer[j + NUM_STATES * INDEX(i.x, 0, i.y)] = stateBuffer[j + NUM_STATES * INDEX(i.x, SIZE_Y - 4, i.y)];
		stateBuffer[j + NUM_STATES * INDEX(i.x, 1, i.y)] = stateBuffer[j + NUM_STATES * INDEX(i.x, SIZE_Y - 3, i.y)];
		stateBuffer[j + NUM_STATES * INDEX(i.x, SIZE_Y - 2, i.y)] = stateBuffer[j + NUM_STATES * INDEX(i.x, 2, i.y)];
		stateBuffer[j + NUM_STATES * INDEX(i.x, SIZE_Y - 1, i.y)] = stateBuffer[j + NUM_STATES * INDEX(i.x, 3, i.y)];
	}
}

__kernel void stateBoundaryPeriodicZ(
	__global real* stateBuffer)
{
	int2 i = (int2)(get_global_id(0), get_global_id(1));
	for (int j = 0; j < NUM_STATES; ++j) {
		stateBuffer[j + NUM_STATES * INDEX(i.x, i.y, 0)] = stateBuffer[j + NUM_STATES * INDEX(i.x, i.y, SIZE_Z - 4)];
		stateBuffer[j + NUM_STATES * INDEX(i.x, i.y, 1)] = stateBuffer[j + NUM_STATES * INDEX(i.x, i.y, SIZE_Z - 3)];
		stateBuffer[j + NUM_STATES * INDEX(i.x, i.y, SIZE_Z - 2)] = stateBuffer[j + NUM_STATES * INDEX(i.x, i.y, 2)];
		stateBuffer[j + NUM_STATES * INDEX(i.x, i.y, SIZE_Z - 1)] = stateBuffer[j + NUM_STATES * INDEX(i.x, i.y, 3)];
	}
}

//mirror

__kernel void stateBoundaryMirrorX(
	__global real* stateBuffer)
{
	int2 i = (int2)(get_global_id(0), get_global_id(1));
	for (int j = 0; j < NUM_STATES; ++j) {
		
		//specific to Euler
		real scale = j == 1 ? -1.f : 1.f;

		stateBuffer[j + NUM_STATES * INDEX(0, i.x, i.y)] = scale * stateBuffer[j + NUM_STATES * INDEX(3, i.x, i.y)];
		stateBuffer[j + NUM_STATES * INDEX(1, i.x, i.y)] = scale * stateBuffer[j + NUM_STATES * INDEX(2, i.x, i.y)];
		stateBuffer[j + NUM_STATES * INDEX(SIZE_X - 1, i.x, i.y)] = scale * stateBuffer[j + NUM_STATES * INDEX(SIZE_X - 4, i.x, i.y)];
		stateBuffer[j + NUM_STATES * INDEX(SIZE_X - 2, i.x, i.y)] = scale * stateBuffer[j + NUM_STATES * INDEX(SIZE_X - 3, i.x, i.y)];
	}
}

__kernel void stateBoundaryMirrorY(
	__global real* stateBuffer)
{
	int2 i = (int2)(get_global_id(0), get_global_id(1));
	for (int j = 0; j < NUM_STATES; ++j) {
		
		//specific to Euler
		real scale = j == 2 ? -1.f : 1.f;
		
		stateBuffer[j + NUM_STATES * INDEX(i.x, 0, i.y)] = scale * stateBuffer[j + NUM_STATES * INDEX(i.x, 3, i.y)];
		stateBuffer[j + NUM_STATES * INDEX(i.x, 1, i.y)] = scale * stateBuffer[j + NUM_STATES * INDEX(i.x, 2, i.y)];
		stateBuffer[j + NUM_STATES * INDEX(i.x, SIZE_Y - 1, i.y)] = scale * stateBuffer[j + NUM_STATES * INDEX(i.x, SIZE_Y - 4, i.y)];
		stateBuffer[j + NUM_STATES * INDEX(i.x, SIZE_Y - 2, i.y)] = scale * stateBuffer[j + NUM_STATES * INDEX(i.x, SIZE_Y - 3, i.y)];
	}
}

__kernel void stateBoundaryMirrorZ(
	__global real* stateBuffer)
{
	int2 i = (int2)(get_global_id(0), get_global_id(1));
	for (int j = 0; j < NUM_STATES; ++j) {
		
		//specific to Euler
		real scale = j == 3 ? -1.f : 1.f;
		
		stateBuffer[j + NUM_STATES * INDEX(i.x, i.y, 0)] = scale * stateBuffer[j + NUM_STATES * INDEX(i.x, i.y, 3)];
		stateBuffer[j + NUM_STATES * INDEX(i.x, i.y, 1)] = scale * stateBuffer[j + NUM_STATES * INDEX(i.x, i.y, 2)];
		stateBuffer[j + NUM_STATES * INDEX(i.x, i.y, SIZE_Z - 1)] = scale * stateBuffer[j + NUM_STATES * INDEX(i.x, i.y, SIZE_Z - 4)];
		stateBuffer[j + NUM_STATES * INDEX(i.x, i.y, SIZE_Z - 2)] = scale * stateBuffer[j + NUM_STATES * INDEX(i.x, i.y, SIZE_Z - 3)];
	}
}

//freeflow

__kernel void stateBoundaryFreeFlowX(
	__global real* stateBuffer)
{
	int2 i = (int2)(get_global_id(0), get_global_id(1));
	for (int j = 0; j < NUM_STATES; ++j) {
		stateBuffer[j + NUM_STATES * INDEX(0, i.x, i.y)] = stateBuffer[j + NUM_STATES * INDEX(1, i.x, i.y)] = stateBuffer[j + NUM_STATES * INDEX(2, i.x, i.y)];
		stateBuffer[j + NUM_STATES * INDEX(SIZE_X - 1, i.x, i.y)] = stateBuffer[j + NUM_STATES * INDEX(SIZE_X - 2, i.x, i.y)] = stateBuffer[j + NUM_STATES * INDEX(SIZE_X - 3, i.x, i.y)];
	}
}

__kernel void stateBoundaryFreeFlowY(
	__global real* stateBuffer)
{
	int2 i = (int2)(get_global_id(0), get_global_id(1));
	for (int j = 0; j < NUM_STATES; ++j) {
		stateBuffer[j + NUM_STATES * INDEX(i.x, 0, i.y)] = stateBuffer[j + NUM_STATES * INDEX(i.x, 1, i.y)] = stateBuffer[j + NUM_STATES * INDEX(i.x, 2, i.y)];
		stateBuffer[j + NUM_STATES * INDEX(i.x, SIZE_Y - 1, i.y)] = stateBuffer[j + NUM_STATES * INDEX(i.x, SIZE_Y - 2, i.y)] = stateBuffer[j + NUM_STATES * INDEX(i.x, SIZE_Y - 3, i.y)];
	}
}

__kernel void stateBoundaryFreeFlowZ(
	__global real* stateBuffer)
{
	int2 i = (int2)(get_global_id(0), get_global_id(1));
	for (int j = 0; j < NUM_STATES; ++j) {
		stateBuffer[j + NUM_STATES * INDEX(i.x, i.y, 0)] = stateBuffer[j + NUM_STATES * INDEX(i.x, i.y, 1)] = stateBuffer[j + NUM_STATES * INDEX(i.x, i.y, 2)];
		stateBuffer[j + NUM_STATES * INDEX(i.x, i.y, SIZE_Z - 1)] = stateBuffer[j + NUM_STATES * INDEX(i.x, i.y, SIZE_Z - 2)] = stateBuffer[j + NUM_STATES * INDEX(i.x, i.y, SIZE_Z - 3)];
	}
}

// the following self-gravitation kernels are specific to Euler but being bound in all solvers 

__kernel void poissonRelax(
	__global real* gravityPotentialBuffer,
	const __global real* stateBuffer,
	int4 repeat)
{
	int4 size = (int4)(SIZE_X, SIZE_Y, SIZE_Z, 0);
	int4 i = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);
	int index = INDEXV(i);

	real sum = 0.f;
	for (int side = 0; side < DIM; ++side) {
		int4 iprev = i;
		int4 inext = i;
		if (repeat[side]) {
			iprev[side] = (iprev[side] + size[side] - 1) % size[side];
			inext[side] = (inext[side] + 1) % size[side];
		} else {
			iprev[side] = max(iprev[side] - 1, 0);
			inext[side] = min(inext[side] + 1, size[side] - 1);
		}
		int indexPrev = INDEXV(iprev);
		int indexNext = INDEXV(inext);
		sum += gravityPotentialBuffer[indexPrev] + gravityPotentialBuffer[indexNext];
	}
	
#define M_PI 3.141592653589793115997963468544185161590576171875f
#define GRAVITY_CONSTANT 1.f		//6.67384e-11 m^3 / (kg s^2)
	real scale = M_PI * GRAVITY_CONSTANT * DX;
#if DIM > 1
	scale *= DY; 
#endif
#if DIM > 2
	scale *= DZ; 
#endif
	real density = stateBuffer[STATE_DENSITY + NUM_STATES * index];
	gravityPotentialBuffer[index] = sum / (2.f * (float)DIM) + scale * density;
}

__kernel void addGravity(
	__global real* stateBuffer,
	const __global real* gravityPotentialBuffer,
	const __global real* dtBuffer)
{
	real dt = dtBuffer[0];
	real4 dt_dx = dt / dx;

	int4 i = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);
	if (i.x < 2 || i.x >= SIZE_X - 2
#if DIM > 1
		|| i.y < 2 || i.y >= SIZE_Y - 2
#endif
#if DIM > 2
		|| i.z < 2 || i.z >= SIZE_Z - 2
#endif
	) {
		return;
	}
	int index = INDEXV(i);

	real density = stateBuffer[STATE_DENSITY + NUM_STATES * index];

	for (int side = 0; side < DIM; ++side) {
		int indexL = index - stepsize[side];
		int indexR = index + stepsize[side];
	
		real gravityGrad = .5f * (gravityPotentialBuffer[indexR] - gravityPotentialBuffer[indexL]);
		
		stateBuffer[side+1 + NUM_STATES * index] -= dt_dx[side] * density * gravityGrad;
		stateBuffer[STATE_ENERGY_TOTAL + NUM_STATES * index] -= dt * density * gravityGrad * stateBuffer[side+1 + NUM_STATES * index];
	}
}

