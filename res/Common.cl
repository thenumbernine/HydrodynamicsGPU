#include "HydroGPU/Shared/Common.h"

constant int4 size = (int4)(SIZE_X, SIZE_Y, SIZE_Z, 0);
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


	//boundary methods


//periodic
//ghost cells copy the opposite side

__kernel void stateBoundaryPeriodicX(
	__global real* buffer,
	int spacing,
	int offset)
{
	int2 i = (int2)(get_global_id(0), get_global_id(1));
	buffer[offset + spacing * INDEX(0, i.x, i.y)] = buffer[offset + spacing * INDEX(SIZE_X - 4, i.x, i.y)];
	buffer[offset + spacing * INDEX(1, i.x, i.y)] = buffer[offset + spacing * INDEX(SIZE_X - 3, i.x, i.y)];
	buffer[offset + spacing * INDEX(SIZE_X - 2, i.x, i.y)] = buffer[offset + spacing * INDEX(2, i.x, i.y)];
	buffer[offset + spacing * INDEX(SIZE_X - 1, i.x, i.y)] = buffer[offset + spacing * INDEX(3, i.x, i.y)];
}

__kernel void stateBoundaryPeriodicY(
	__global real* buffer,
	int spacing,
	int offset)
{
	int2 i = (int2)(get_global_id(0), get_global_id(1));
	buffer[offset + spacing * INDEX(i.x, 0, i.y)] = buffer[offset + spacing * INDEX(i.x, SIZE_Y - 4, i.y)];
	buffer[offset + spacing * INDEX(i.x, 1, i.y)] = buffer[offset + spacing * INDEX(i.x, SIZE_Y - 3, i.y)];
	buffer[offset + spacing * INDEX(i.x, SIZE_Y - 2, i.y)] = buffer[offset + spacing * INDEX(i.x, 2, i.y)];
	buffer[offset + spacing * INDEX(i.x, SIZE_Y - 1, i.y)] = buffer[offset + spacing * INDEX(i.x, 3, i.y)];
}

__kernel void stateBoundaryPeriodicZ(
	__global real* buffer,
	int spacing,
	int offset)
{
	int2 i = (int2)(get_global_id(0), get_global_id(1));
	buffer[offset + spacing * INDEX(i.x, i.y, 0)] = buffer[offset + spacing * INDEX(i.x, i.y, SIZE_Z - 4)];
	buffer[offset + spacing * INDEX(i.x, i.y, 1)] = buffer[offset + spacing * INDEX(i.x, i.y, SIZE_Z - 3)];
	buffer[offset + spacing * INDEX(i.x, i.y, SIZE_Z - 2)] = buffer[offset + spacing * INDEX(i.x, i.y, 2)];
	buffer[offset + spacing * INDEX(i.x, i.y, SIZE_Z - 1)] = buffer[offset + spacing * INDEX(i.x, i.y, 3)];
}

//mirror
//ghost cells mirror the next adjacent cells

__kernel void stateBoundaryMirrorX(
	__global real* buffer,
	int spacing,
	int offset)
{
	int2 i = (int2)(get_global_id(0), get_global_id(1));
	buffer[offset + spacing * INDEX(0, i.x, i.y)] = buffer[offset + spacing * INDEX(3, i.x, i.y)];
	buffer[offset + spacing * INDEX(1, i.x, i.y)] = buffer[offset + spacing * INDEX(2, i.x, i.y)];
	buffer[offset + spacing * INDEX(SIZE_X - 1, i.x, i.y)] = buffer[offset + spacing * INDEX(SIZE_X - 4, i.x, i.y)];
	buffer[offset + spacing * INDEX(SIZE_X - 2, i.x, i.y)] = buffer[offset + spacing * INDEX(SIZE_X - 3, i.x, i.y)];
}

__kernel void stateBoundaryMirrorY(
	__global real* buffer,
	int spacing,
	int offset)
{
	int2 i = (int2)(get_global_id(0), get_global_id(1));
	buffer[offset + spacing * INDEX(i.x, 0, i.y)] = buffer[offset + spacing * INDEX(i.x, 3, i.y)];
	buffer[offset + spacing * INDEX(i.x, 1, i.y)] = buffer[offset + spacing * INDEX(i.x, 2, i.y)];
	buffer[offset + spacing * INDEX(i.x, SIZE_Y - 1, i.y)] = buffer[offset + spacing * INDEX(i.x, SIZE_Y - 4, i.y)];
	buffer[offset + spacing * INDEX(i.x, SIZE_Y - 2, i.y)] = buffer[offset + spacing * INDEX(i.x, SIZE_Y - 3, i.y)];
}

__kernel void stateBoundaryMirrorZ(
	__global real* buffer,
	int spacing,
	int offset)
{
	int2 i = (int2)(get_global_id(0), get_global_id(1));
	buffer[offset + spacing * INDEX(i.x, i.y, 0)] = buffer[offset + spacing * INDEX(i.x, i.y, 3)];
	buffer[offset + spacing * INDEX(i.x, i.y, 1)] = buffer[offset + spacing * INDEX(i.x, i.y, 2)];
	buffer[offset + spacing * INDEX(i.x, i.y, SIZE_Z - 1)] = buffer[offset + spacing * INDEX(i.x, i.y, SIZE_Z - 4)];
	buffer[offset + spacing * INDEX(i.x, i.y, SIZE_Z - 2)] = buffer[offset + spacing * INDEX(i.x, i.y, SIZE_Z - 3)];
}

//reflect
//ghost cells are negatives of the mirror of the next adjacent cells

__kernel void stateBoundaryReflectX(
	__global real* buffer,
	int spacing,
	int offset)
{
	int2 i = (int2)(get_global_id(0), get_global_id(1));
	buffer[offset + spacing * INDEX(0, i.x, i.y)] = -buffer[offset + spacing * INDEX(3, i.x, i.y)];
	buffer[offset + spacing * INDEX(1, i.x, i.y)] = -buffer[offset + spacing * INDEX(2, i.x, i.y)];
	buffer[offset + spacing * INDEX(SIZE_X - 1, i.x, i.y)] = -buffer[offset + spacing * INDEX(SIZE_X - 4, i.x, i.y)];
	buffer[offset + spacing * INDEX(SIZE_X - 2, i.x, i.y)] = -buffer[offset + spacing * INDEX(SIZE_X - 3, i.x, i.y)];
}

__kernel void stateBoundaryReflectY(
	__global real* buffer,
	int spacing,
	int offset)
{
	int2 i = (int2)(get_global_id(0), get_global_id(1));
	buffer[offset + spacing * INDEX(i.x, 0, i.y)] = -buffer[offset + spacing * INDEX(i.x, 3, i.y)];
	buffer[offset + spacing * INDEX(i.x, 1, i.y)] = -buffer[offset + spacing * INDEX(i.x, 2, i.y)];
	buffer[offset + spacing * INDEX(i.x, SIZE_Y - 1, i.y)] = -buffer[offset + spacing * INDEX(i.x, SIZE_Y - 4, i.y)];
	buffer[offset + spacing * INDEX(i.x, SIZE_Y - 2, i.y)] = -buffer[offset + spacing * INDEX(i.x, SIZE_Y - 3, i.y)];
}

__kernel void stateBoundaryReflectZ(
	__global real* buffer,
	int spacing,
	int offset)
{
	int2 i = (int2)(get_global_id(0), get_global_id(1));
	buffer[offset + spacing * INDEX(i.x, i.y, 0)] = -buffer[offset + spacing * INDEX(i.x, i.y, 3)];
	buffer[offset + spacing * INDEX(i.x, i.y, 1)] = -buffer[offset + spacing * INDEX(i.x, i.y, 2)];
	buffer[offset + spacing * INDEX(i.x, i.y, SIZE_Z - 1)] = -buffer[offset + spacing * INDEX(i.x, i.y, SIZE_Z - 4)];
	buffer[offset + spacing * INDEX(i.x, i.y, SIZE_Z - 2)] = -buffer[offset + spacing * INDEX(i.x, i.y, SIZE_Z - 3)];
}

//freeflow
//ghost cells copy the next adjacent cell

__kernel void stateBoundaryFreeFlowX(
	__global real* buffer,
	int spacing,
	int offset)
{
	int2 i = (int2)(get_global_id(0), get_global_id(1));
	buffer[offset + spacing * INDEX(0, i.x, i.y)] = buffer[offset + spacing * INDEX(1, i.x, i.y)] = buffer[offset + spacing * INDEX(2, i.x, i.y)];
	buffer[offset + spacing * INDEX(SIZE_X - 1, i.x, i.y)] = buffer[offset + spacing * INDEX(SIZE_X - 2, i.x, i.y)] = buffer[offset + spacing * INDEX(SIZE_X - 3, i.x, i.y)];
}

__kernel void stateBoundaryFreeFlowY(
	__global real* buffer,
	int spacing,
	int offset)
{
	int2 i = (int2)(get_global_id(0), get_global_id(1));
	buffer[offset + spacing * INDEX(i.x, 0, i.y)] = buffer[offset + spacing * INDEX(i.x, 1, i.y)] = buffer[offset + spacing * INDEX(i.x, 2, i.y)];
	buffer[offset + spacing * INDEX(i.x, SIZE_Y - 1, i.y)] = buffer[offset + spacing * INDEX(i.x, SIZE_Y - 2, i.y)] = buffer[offset + spacing * INDEX(i.x, SIZE_Y - 3, i.y)];
}

__kernel void stateBoundaryFreeFlowZ(
	__global real* buffer,
	int spacing,
	int offset)
{
	int2 i = (int2)(get_global_id(0), get_global_id(1));
	buffer[offset + spacing * INDEX(i.x, i.y, 0)] = buffer[offset + spacing * INDEX(i.x, i.y, 1)] = buffer[offset + spacing * INDEX(i.x, i.y, 2)];
	buffer[offset + spacing * INDEX(i.x, i.y, SIZE_Z - 1)] = buffer[offset + spacing * INDEX(i.x, i.y, SIZE_Z - 2)] = buffer[offset + spacing * INDEX(i.x, i.y, SIZE_Z - 3)];
}


	//integration methods


__kernel void forwardEulerIntegrate(
	__global real* stateBuffer,
	const __global real* derivBuffer,
	const __global real* dtBuffer)
{
	real dt = dtBuffer[0];
	
	int4 i = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);
	if (i.x < 2 || i.x >= SIZE_X - 2 
#if DIM > 1
		|| i.y < 2 || i.y >= SIZE_Y - 2 
#if DIM > 2
		|| i.z < 2 || i.z >= SIZE_Z - 2
#endif
#endif
	) {
		return;
	}
	int index = INDEXV(i);

	__global real* state = stateBuffer + NUM_STATES * index;
	const __global real* deriv = derivBuffer + NUM_STATES * index;
	
	for (int j = 0; j < NUM_STATES; ++j) {
		state[j] += dt * deriv[j];
	}
}

