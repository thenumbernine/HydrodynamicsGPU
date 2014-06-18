#include "HydroGPU/Shared/Common3D.h"

real8 mirror(real8 state, int mirrorDim);

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
	stateBuffer[INDEX(i.y, 0, i.x)] = stateBuffer[INDEX(i.y, size.y - 4, i.x)];
	stateBuffer[INDEX(i.y, 1, i.x)] = stateBuffer[INDEX(i.y, size.y - 3, i.x)];
	stateBuffer[INDEX(i.y, size.y - 2, i.x)] = stateBuffer[INDEX(i.y, 2, i.x)];
	stateBuffer[INDEX(i.y, size.y - 1, i.x)] = stateBuffer[INDEX(i.y, 3, i.x)];
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

real8 mirror(real8 state, int mirrorDim) {
	state[mirrorDim] = -state[mirrorDim];
	return state;
}

__kernel void stateBoundaryMirrorX(
	__global real8* stateBuffer,
	int3 size)
{
	int2 i = (int2)(get_global_id(0), get_global_id(1));
	stateBuffer[INDEX(0, i.x, i.y)] = mirror(stateBuffer[INDEX(3, i.x, i.y)], 1);
	stateBuffer[INDEX(1, i.x, i.y)] = mirror(stateBuffer[INDEX(2, i.x, i.y)], 1);
	stateBuffer[INDEX(size.x - 1, i.x, i.y)] = mirror(stateBuffer[INDEX(size.x - 4, i.x, i.y)], 1);
	stateBuffer[INDEX(size.x - 2, i.x, i.y)] = mirror(stateBuffer[INDEX(size.x - 3, i.x, i.y)], 1);
}

__kernel void stateBoundaryMirrorY(
	__global real8* stateBuffer,
	int3 size)
{
	int2 i = (int2)(get_global_id(0), get_global_id(1));
	stateBuffer[INDEX(i.y, 0, i.x)] = mirror(stateBuffer[INDEX(i.y, 3, i.x)], 2);
	stateBuffer[INDEX(i.y, 1, i.x)] = mirror(stateBuffer[INDEX(i.y, 2, i.x)], 2);
	stateBuffer[INDEX(i.y, size.y - 1, i.x)] = mirror(stateBuffer[INDEX(i.y, size.y - 4, i.x)], 2);
	stateBuffer[INDEX(i.y, size.y - 2, i.x)] = mirror(stateBuffer[INDEX(i.y, size.y - 3, i.x)], 2);
}

__kernel void stateBoundaryMirrorZ(
	__global real8* stateBuffer,
	int3 size)
{
	int2 i = (int2)(get_global_id(0), get_global_id(1));
	stateBuffer[INDEX(i.x, i.y, 0)] = mirror(stateBuffer[INDEX(i.x, i.y, 3)], 3);
	stateBuffer[INDEX(i.x, i.y, 1)] = mirror(stateBuffer[INDEX(i.x, i.y, 2)], 3);
	stateBuffer[INDEX(i.x, i.y, size.z - 1)] = mirror(stateBuffer[INDEX(i.x, i.y, size.z - 4)], 3);
	stateBuffer[INDEX(i.x, i.y, size.z - 2)] = mirror(stateBuffer[INDEX(i.x, i.y, size.z - 3)], 3);
}

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
	stateBuffer[INDEX(i.y, 0, i.x)] = stateBuffer[INDEX(i.y, 2, i.x)] = stateBuffer[INDEX(i.y, 2, i.x)];
	stateBuffer[INDEX(i.y, size.y - 1, i.x)] = stateBuffer[INDEX(i.y, size.y - 3, i.x)] = stateBuffer[INDEX(i.y, size.y - 3, i.x)];
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

	float4 color = read_imagef(gradientTex, CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_REPEAT | CLK_FILTER_LINEAR, value);
	write_imagef(fluidTex, (int4)(i.x, i.y, i.z, 0), color.bgra);
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
	for (int side = 0; side < 3; ++side) {
		int3 iprev = i;
		int3 inext = i;
	/*	
		if (repeat[side]) {
			iprev[side] = (iprev[side] + size[side] - 1) % size[side];
			inext[side] = (inext[side] + 1) % size[side];
		} else {
	*/
			iprev[side] = max(iprev[side] - 1, 0);
			inext[side] = min(inext[side] + 1, size[side] - 1);
	//	}
		int indexPrev = INDEXV(iprev);
		int indexNext = INDEXV(inext);
		sum += gravityPotentialBuffer[indexPrev] + gravityPotentialBuffer[indexNext];
	}
	
#define M_PI 3.141592653589793115997963468544185161590576171875f
#define GRAVITY_CONSTANT 1.f		//6.67384e-11 m^3 / (kg s^2)
	real scale = M_PI * GRAVITY_CONSTANT * dx.x * dx.y * dx.z;

	gravityPotentialBuffer[index] = sum / 6.f + scale * stateBuffer[index].s0;
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
