#include "HydroGPU/Shared/Common2D.h"

real8 mirrorStateX(real8 state);
real8 mirrorStateY(real8 state);

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
	__global real8* stateBuffer)
{
	int i = get_global_id(0);
	stateBuffer[INDEX(0, i)] = stateBuffer[INDEX(SIZE_X - 4, i)];
	stateBuffer[INDEX(1, i)] = stateBuffer[INDEX(SIZE_X - 3, i)];
	stateBuffer[INDEX(SIZE_X - 2, i)] = stateBuffer[INDEX(2, i)];
	stateBuffer[INDEX(SIZE_X - 1, i)] = stateBuffer[INDEX(3, i)];
}

__kernel void stateBoundaryPeriodicY(
	__global real8* stateBuffer)
{
	int i = get_global_id(0);
	stateBuffer[INDEX(i, 0)] = stateBuffer[INDEX(i, SIZE_Y - 4)];
	stateBuffer[INDEX(i, 1)] = stateBuffer[INDEX(i, SIZE_Y - 3)];
	stateBuffer[INDEX(i, SIZE_Y - 2)] = stateBuffer[INDEX(i, 2)];
	stateBuffer[INDEX(i, SIZE_Y - 1)] = stateBuffer[INDEX(i, 3)];
}

//mirror

real8 mirrorStateX(real8 state) { state.s1 = -state.s1; return state; }
real8 mirrorStateY(real8 state) { state.s2 = -state.s2; return state; }

__kernel void stateBoundaryMirrorX(
	__global real8* stateBuffer)
{
	int i = get_global_id(0);
	stateBuffer[INDEX(0, i)] = mirrorStateX(stateBuffer[INDEX(3, i)]);
	stateBuffer[INDEX(1, i)] = mirrorStateX(stateBuffer[INDEX(2, i)]);
	stateBuffer[INDEX(SIZE_X - 1, i)] = mirrorStateX(stateBuffer[INDEX(SIZE_X - 4, i)]);
	stateBuffer[INDEX(SIZE_X - 2, i)] = mirrorStateX(stateBuffer[INDEX(SIZE_X - 3, i)]);	
}

__kernel void stateBoundaryMirrorY(
	__global real8* stateBuffer)
{
	int i = get_global_id(0);
	stateBuffer[INDEX(i, 0)] = mirrorStateY(stateBuffer[INDEX(i, 3)]);
	stateBuffer[INDEX(i, 1)] = mirrorStateY(stateBuffer[INDEX(i, 2)]);
	stateBuffer[INDEX(i, SIZE_Y - 1)] = mirrorStateY(stateBuffer[INDEX(i, SIZE_Y - 4)]);
	stateBuffer[INDEX(i, SIZE_Y - 2)] = mirrorStateY(stateBuffer[INDEX(i, SIZE_Y - 3)]);
}

//freeflow

__kernel void stateBoundaryFreeFlowX(
	__global real8* stateBuffer)
{
	int i = get_global_id(0);
	stateBuffer[INDEX(0, i)] = stateBuffer[INDEX(1, i)] = stateBuffer[INDEX(2, i)];
	stateBuffer[INDEX(SIZE_X - 1, i)] = stateBuffer[INDEX(SIZE_X - 2, i)] = stateBuffer[INDEX(SIZE_X - 3, i)];
}

__kernel void stateBoundaryFreeFlowY(
	__global real8* stateBuffer)
{
	int i = get_global_id(0);
	stateBuffer[INDEX(i, 0)] = stateBuffer[INDEX(i, 1)] = stateBuffer[INDEX(i, 2)];
	stateBuffer[INDEX(i, SIZE_Y - 1)] = stateBuffer[INDEX(i, SIZE_Y - 2)] = stateBuffer[INDEX(i, SIZE_Y - 3)];
}

__kernel void convertToTex(
	const __global real8* stateBuffer,
	const __global real* gravityPotentialBuffer,
	__write_only image2d_t fluidTex,
	__read_only image1d_t gradientTex,
	int displayMethod,
	float displayScale)
{
	int2 i = (int2)(get_global_id(0), get_global_id(1));
	int index = INDEXV(i);
	
	real8 state = stateBuffer[index];
	
#if DIM == 1
	real density = state.s0;
	real4 velocity = (real4)(state.s1, state.s2, state.s3, 0.f) / density;;
	real velocityMagn = length(velocity);
	real energyTotal = state.s4 / density;
	real energyKinetic = .5f * dot(velocity, velocity);
	real energyPotential = gravityPotentialBuffer[index];
	real energyInternal = energyTotal - energyKinetic - energyPotential;
	float4 color = (float4)(density, velocityMagn, energyInternal, 0.f) * displayScale;
	write_imagef(fluidTex, i, color);
#elif DIM == 2

	real value;
	switch (displayMethod) {
	case DISPLAY_DENSITY:	//density
		value = state.s0;
		break;
	case DISPLAY_VELOCITY:	//velocity
		value = length(state.s12) / state.s0;
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
	case DISPLAY_MAGNETISM:
		value = length(state.s567);
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
	write_imagef(fluidTex, i, color.bgra);
#endif
}

__kernel void poissonRelax(
	__global real* gravityPotentialBuffer,
	const __global real8* stateBuffer,
	int2 repeat)
{
	int2 size = (int2)(SIZE_X, SIZE_Y);
	int2 i = (int2)(get_global_id(0), get_global_id(1));
	int index = INDEXV(i);

	real sum = 0.f;
	for (int side = 0; side < DIM; ++side) {
		int2 iprev = i;
		int2 inext = i;
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

	gravityPotentialBuffer[index] = sum / (2.f * (float)DIM) + scale * stateBuffer[index].s0;
}

__kernel void addGravity(
	__global real8* stateBuffer,
	const __global real* gravityPotentialBuffer,
	const __global real* dtBuffer)
{
	real dt = dtBuffer[0];
	real2 dx = (real2)(DX, DY);
	real2 dt_dx = dt / dx;

	int2 i = (int2)(get_global_id(0), get_global_id(1));
	if (i.x < 2 || i.x >= SIZE_X - 2
#if DIM > 1
		|| i.y < 2 || i.y >= SIZE_Y - 2
#endif
	) {
		return;
	}
	int index = INDEXV(i);

	real8 state = stateBuffer[index];
	real density = state.x;

	for (int side = 0; side < DIM; ++side) {
		int2 iPrev = i;
		--iPrev[side];
		int indexPrev = iPrev.x + SIZE_X * iPrev.y;
		
		int2 iNext = i;
		++iNext[side];
		int indexNext = iNext.x + SIZE_X * iNext.y;	
	
		real gravityGrad = .5f * (gravityPotentialBuffer[indexNext] - gravityPotentialBuffer[indexPrev]);
		
		state[side+1] -= dt_dx[side] * density * gravityGrad;
		state.s4 -= dt * density * gravityGrad * state[side+1];
	}

	stateBuffer[index] = state;
}

__kernel void addDrop(
	__global real8* stateBuffer,
	real2 xmin,
	real2 xmax,
	const __global real* dt,
	real2 pos,
	real2 sourceVelocity)
{
#if 0
	int2 i = (int2)(get_global_id(0), get_global_id(1));
	if (i.x >= SIZE_X || i.y >= SIZE_Y) return;

	int index = i.x + SIZE_X * i.y;
	real8 state = stateBuffer[index];

	const float dropRadius = .02f;
	const float densityMagnitude = 50.f;
	const float velocityMagnitude = 10000.f;
	const float energyInternalMagnitude = 0.f;
	
	real cellPosX = (real)i.x / (real)SIZE_X * (xmax.x - xmin.x) + xmin.x;
	real cellPosY = (real)i.y / (real)SIZE_Y * (xmax.y - xmin.y) + xmin.y;
	real2 cellPos = (real2)(cellPosX, cellPosY);
	real2 dx = (cellPos - pos) / dropRadius;
	float rSq = dot(dx, dx);
	float falloff = exp(-rSq);
	
	real density = state.x;
	real2 velocity = state.yz / density;
	real energyTotal = state.w / density;
	real energyKinetic = .5f * dot(velocity, velocity);
	real energyInternal = energyTotal - energyKinetic;

	density += *dt * densityMagnitude * falloff;
	velocity += *dt * sourceVelocity * (falloff * velocityMagnitude);
	energyInternal += *dt * energyInternalMagnitude * falloff;

	energyKinetic = .5f * dot(velocity, velocity);
	energyTotal = energyInternal + energyKinetic;

	stateBuffer[index] = (real4)(1.f, velocity.x, velocity.y, energyTotal) * density;
#endif
}

__kernel void addSource(
	__global real8* stateBuffer,
	real2 xmin,
	real2 xmax,
	const __global real* dt)
{
#if 0
	int2 i = (int2)(get_global_id(0), get_global_id(1));
	if (i.x >= SIZE_X || i.y >= SIZE_Y) return;

	const float radius = .05f;

	real cellPosX = (real)i.x / (real)SIZE_X * (xmax.x - xmin.x) + xmin.x;
	real cellPosY = (real)i.y / (real)SIZE_Y * (xmax.y - xmin.y) + xmin.y;
	real2 cellPos = (real2)(cellPosX, cellPosY);
	real2 sourcePos = (real2)(xmin.x, .5f * (xmax.y + xmin.y));
	real2 dx = (cellPos - sourcePos) / radius;
	real rSq = dot(dx,dx);
	real falloff = exp(-rSq);

	int index = i.x + SIZE_X * i.y;
	real8 state = stateBuffer[index];
	real density = state.x;
	real2 velocity = state.yz / density;
	real energyTotal = state.w / density;
	real energyKinetic = .5f * dot(velocity, velocity);
	real energyInternal = energyTotal - energyKinetic;

	const float velocityMagnitude = 10.f;
	velocity.x += velocityMagnitude * falloff * *dt;

	energyKinetic = .5f * dot(velocity, velocity);
	energyTotal = energyInternal + energyKinetic;

	stateBuffer[index] = (real4)(1.f, velocity.x, velocity.y, energyTotal) * density;
#endif
}


