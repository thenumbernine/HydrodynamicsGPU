#include "HydroGPU/Shared/Common3D.h"

real8 mirrorStateX(real8 state);
real8 mirrorStateY(real8 state);
real8 mirrorStateZ(real8 state);

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
	__global real8* stateBuffer,
	int3 size)
{
	int2 i = (int2)(get_global_id(0), get_global_id(1));
	stateBuffer[INDEX(0, i.x, i.y)] = stateBuffer[INDEX(size.x - 4, i.x, i.y)];
	stateBuffer[INDEX(1, i.x, i.y)] = stateBuffer[INDEX(size.x - 3, i.x, i.y)];
	stateBuffer[INDEX(size.x - 2, i.x, i.y)] = stateBuffer[INDEX(2, i.x, i.y)];
	stateBuffer[INDEX(size.x - 1, i.x, i.y)] = stateBuffer[INDEX(3, i.x, i.y)];
}

__kernel void stateBoundaryPeriodicY(
	__global real8* stateBuffer,
	int3 size)
{
	int2 i = (int2)(get_global_id(0), get_global_id(1));
	stateBuffer[INDEX(i.x, 0, i.y)] = stateBuffer[INDEX(i.x, size.y - 4, i.y)];
	stateBuffer[INDEX(i.x, 1, i.y)] = stateBuffer[INDEX(i.x, size.y - 3, i.y)];
	stateBuffer[INDEX(i.x, size.y - 2, i.y)] = stateBuffer[INDEX(i.x, 2, i.y)];
	stateBuffer[INDEX(i.x, size.y - 1, i.y)] = stateBuffer[INDEX(i.x, 3, i.y)];
}

__kernel void stateBoundaryPeriodicZ(
	__global real8* stateBuffer,
	int3 size)
{
	int2 i = (int2)(get_global_id(0), get_global_id(1));
	stateBuffer[INDEX(i.x, i.y, 0)] = stateBuffer[INDEX(i.x, i.y, size.z - 4)];
	stateBuffer[INDEX(i.x, i.y, 1)] = stateBuffer[INDEX(i.x, i.y, size.z - 3)];
	stateBuffer[INDEX(i.x, i.y, size.z - 2)] = stateBuffer[INDEX(i.x, i.y, 2)];
	stateBuffer[INDEX(i.x, i.y, size.z - 1)] = stateBuffer[INDEX(i.x, i.y, 3)];
}

//mirror

real8 mirrorStateX(real8 state) { state.s1 = -state.s1; return state; }
real8 mirrorStateY(real8 state) { state.s2 = -state.s2; return state; }
real8 mirrorStateZ(real8 state) { state.s3 = -state.s3; return state; }

__kernel void stateBoundaryMirrorX(
	__global real8* stateBuffer,
	int3 size)
{
	int2 i = (int2)(get_global_id(0), get_global_id(1));
	stateBuffer[INDEX(0, i.x, i.y)] = mirrorStateX(stateBuffer[INDEX(3, i.x, i.y)]);
	stateBuffer[INDEX(1, i.x, i.y)] = mirrorStateX(stateBuffer[INDEX(2, i.x, i.y)]);
	stateBuffer[INDEX(size.x - 1, i.x, i.y)] = mirrorStateX(stateBuffer[INDEX(size.x - 4, i.x, i.y)]);
	stateBuffer[INDEX(size.x - 2, i.x, i.y)] = mirrorStateX(stateBuffer[INDEX(size.x - 3, i.x, i.y)]);
}

__kernel void stateBoundaryMirrorY(
	__global real8* stateBuffer,
	int3 size)
{
	int2 i = (int2)(get_global_id(0), get_global_id(1));
	stateBuffer[INDEX(i.x, 0, i.y)] = mirrorStateY(stateBuffer[INDEX(i.x, 3, i.y)]);
	stateBuffer[INDEX(i.x, 1, i.y)] = mirrorStateY(stateBuffer[INDEX(i.x, 2, i.y)]);
	stateBuffer[INDEX(i.x, size.y - 1, i.y)] = mirrorStateY(stateBuffer[INDEX(i.x, size.y - 4, i.y)]);
	stateBuffer[INDEX(i.x, size.y - 2, i.y)] = mirrorStateY(stateBuffer[INDEX(i.x, size.y - 3, i.y)]);
}

__kernel void stateBoundaryMirrorZ(
	__global real8* stateBuffer,
	int3 size)
{
	int2 i = (int2)(get_global_id(0), get_global_id(1));
	stateBuffer[INDEX(i.x, i.y, 0)] = mirrorStateZ(stateBuffer[INDEX(i.x, i.y, 3)]);
	stateBuffer[INDEX(i.x, i.y, 1)] = mirrorStateZ(stateBuffer[INDEX(i.x, i.y, 2)]);
	stateBuffer[INDEX(i.x, i.y, size.z - 1)] = mirrorStateZ(stateBuffer[INDEX(i.x, i.y, size.z - 4)]);
	stateBuffer[INDEX(i.x, i.y, size.z - 2)] = mirrorStateZ(stateBuffer[INDEX(i.x, i.y, size.z - 3)]);
}

//freeflow

__kernel void stateBoundaryFreeFlowX(
	__global real8* stateBuffer,
	int3 size)
{
	int2 i = (int2)(get_global_id(0), get_global_id(1));
	stateBuffer[INDEX(0, i.x, i.y)] = stateBuffer[INDEX(2, i.x, i.y)] = stateBuffer[INDEX(2, i.x, i.y)];
	stateBuffer[INDEX(size.x - 1, i.x, i.y)] = stateBuffer[INDEX(size.x - 3, i.x, i.y)] = stateBuffer[INDEX(size.x - 3, i.x, i.y)];
}

__kernel void stateBoundaryFreeFlowY(
	__global real8* stateBuffer,
	int3 size)
{
	int2 i = (int2)(get_global_id(0), get_global_id(1));
	stateBuffer[INDEX(i.x, 0, i.y)] = stateBuffer[INDEX(i.x, 2, i.y)] = stateBuffer[INDEX(i.x, 2, i.y)];
	stateBuffer[INDEX(i.x, size.y - 1, i.y)] = stateBuffer[INDEX(i.x, size.y - 3, i.y)] = stateBuffer[INDEX(i.x, size.y - 3, i.y)];
}

__kernel void stateBoundaryFreeFlowZ(
	__global real8* stateBuffer,
	int3 size)
{
	int2 i = (int2)(get_global_id(0), get_global_id(1));
	stateBuffer[INDEX(i.x, i.y, 0)] = stateBuffer[INDEX(i.x, i.y, 2)] = stateBuffer[INDEX(i.x, i.y, 2)];
	stateBuffer[INDEX(i.x, i.y, size.z - 1)] = stateBuffer[INDEX(i.x, i.y, size.z - 3)] = stateBuffer[INDEX(i.x, i.y, size.z - 3)];
}

__kernel void convertToTex(
	const __global real8* stateBuffer,
	const __global real* gravityPotentialBuffer,
	int3 size,
	__write_only image3d_t fluidTex,
	__read_only image1d_t gradientTex,
	int displayMethod,
	float displayScale)
{
	int3 i = (int3)(get_global_id(0), get_global_id(1), get_global_id(2));
	int index = INDEXV(i);
	
	real8 state = stateBuffer[index];

#if DIM == 1
	real density = state.s0;
	real velocity = length(state.s123) / density;
	real energyTotal = state.s4 / density;
	real energyKinetic = .5f * velocity * velocity;
	real energyPotential = gravityPotentialBuffer[index];
	real energyInternal = energyTotal - energyKinetic - energyPotential;
	float4 color = (float4)(density, velocity, energyInternal, 0.f) * displayScale;
#else
	real value;
	switch (displayMethod) {
	case DISPLAY_DENSITY:	//density
		value = state.s0;
		break;
	case DISPLAY_VELOCITY:	//velocity
		value = length(state.s123) / state.s0;
		break;
	case DISPLAY_PRESSURE:	//pressure
		{
			real density = state.s0;
			real energyTotal = state.s4 / density;
			real energyKinetic = .5f * dot(state.s123, state.s123) / (density * density);
			real energyPotential = gravityPotentialBuffer[index];
			real energyInternal = energyTotal - energyKinetic - energyPotential;
			value = (GAMMA - 1.f) * energyInternal * density;
		}
		break;
	case DISPLAY_GRAVITY_POTENTIAL:
		value = gravityPotentialBuffer[index];
		break;
	default:
		value = .5f;
		break;
	}
	value *= displayScale;

	float4 color = read_imagef(gradientTex, CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_REPEAT | CLK_FILTER_LINEAR, value).bgra;
#endif
	write_imagef(fluidTex, (int4)(i.x, i.y, i.z, 0), color);
}

__kernel void poissonRelax(
	__global real* gravityPotentialBuffer,
	const __global real8* stateBuffer,
	int3 size,
	real3 dx,
	int3 repeat)
{
	int3 i = (int3)(get_global_id(0), get_global_id(1), get_global_id(2));
	int index = INDEXV(i);

	real sum = 0.f;
	for (int side = 0; side < DIM; ++side) {
		int3 iprev = i;
		int3 inext = i;
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
	real scale = M_PI * GRAVITY_CONSTANT * dx.x * dx.y * dx.z;
#if DIM > 1
	scale *= dx.y;
#if DIM > 2
	scale *= dx.z;
#endif
#endif
	gravityPotentialBuffer[index] = sum / (2.f * (float)DIM) + scale * stateBuffer[index].s0;
}

__kernel void addGravity(
	__global real8* stateBuffer,
	const __global real* gravityPotentialBuffer,
	int3 size,
	real3 dx,
	const __global real* dtBuffer)
{
	real dt = dtBuffer[0];
	real3 dt_dx = dt / dx;

	int3 i = (int3)(get_global_id(0), get_global_id(1), get_global_id(2));
	if (i.x < 2 || i.y < 2 || i.z < 2 || i.x >= size.x - 2 || i.y >= size.y - 2 || i.z >= size.z - 2) return;
	int index = INDEXV(i);

	real8 state = stateBuffer[index];
	real density = state.x;

	for (int side = 0; side < 3; ++side) {
		int3 iPrev = i;
		--iPrev[side];
		int indexPrev = INDEXV(iPrev);
		
		int3 iNext = i;
		++iNext[side];
		int indexNext = INDEXV(iNext);
	
		real gravityGrad = .5f * (gravityPotentialBuffer[indexNext] - gravityPotentialBuffer[indexPrev]);
		
		state[side+1] -= dt_dx[side] * density * gravityGrad;
		state.s4 -= dt * density * gravityGrad * state[side+1];
	}

	stateBuffer[index] = state;
}
