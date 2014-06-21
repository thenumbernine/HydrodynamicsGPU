#include "HydroGPU/Shared/Common2D.h"

real4 mirrorStateX(real4 state);
real4 mirrorStateY(real4 state);

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
	int2 size)
{
	int i = get_global_id(0);
	stateBuffer[INDEX(0, i)].s0123 = stateBuffer[INDEX(size.x - 4, i)].s0123;
	stateBuffer[INDEX(1, i)].s0123 = stateBuffer[INDEX(size.x - 3, i)].s0123;
	stateBuffer[INDEX(size.x - 2, i)].s0123 = stateBuffer[INDEX(2, i)].s0123;
	stateBuffer[INDEX(size.x - 1, i)].s0123 = stateBuffer[INDEX(3, i)].s0123;
}

__kernel void stateBoundaryPeriodicY(
	__global real8* stateBuffer,
	int2 size)
{
	int i = get_global_id(0);
	stateBuffer[INDEX(i, 0)].s0123 = stateBuffer[INDEX(i, size.y - 4)].s0123;
	stateBuffer[INDEX(i, 1)].s0123 = stateBuffer[INDEX(i, size.y - 3)].s0123;
	stateBuffer[INDEX(i, size.y - 2)].s0123 = stateBuffer[INDEX(i, 2)].s0123;
	stateBuffer[INDEX(i, size.y - 1)].s0123 = stateBuffer[INDEX(i, 3)].s0123;
}

//mirror

real4 mirrorStateX(real4 state) { state.s1 = -state.s1; return state; }
real4 mirrorStateY(real4 state) { state.s2 = -state.s2; return state; }

__kernel void stateBoundaryMirrorX(
	__global real8* stateBuffer,
	int2 size)
{
	int i = get_global_id(0);
	stateBuffer[INDEX(0, i)].s0123 = mirrorStateX(stateBuffer[INDEX(3, i)].s0123);
	stateBuffer[INDEX(1, i)].s0123 = mirrorStateX(stateBuffer[INDEX(2, i)].s0123);
	stateBuffer[INDEX(size.x - 1, i)].s0123 = mirrorStateX(stateBuffer[INDEX(size.x - 4, i)].s0123);
	stateBuffer[INDEX(size.x - 2, i)].s0123 = mirrorStateX(stateBuffer[INDEX(size.x - 3, i)].s0123);	
}

__kernel void stateBoundaryMirrorY(
	__global real8* stateBuffer,
	int2 size)
{
	int i = get_global_id(0);
	stateBuffer[INDEX(i, 0)].s0123 = mirrorStateY(stateBuffer[INDEX(i, 3)].s0123);
	stateBuffer[INDEX(i, 1)].s0123 = mirrorStateY(stateBuffer[INDEX(i, 2)].s0123);
	stateBuffer[INDEX(i, size.y - 1)].s0123 = mirrorStateY(stateBuffer[INDEX(i, size.y - 4)].s0123);
	stateBuffer[INDEX(i, size.y - 2)].s0123 = mirrorStateY(stateBuffer[INDEX(i, size.y - 3)].s0123);
}

//freeflow

__kernel void stateBoundaryFreeFlowX(
	__global real8* stateBuffer,
	int2 size)
{
	int i = get_global_id(0);
	stateBuffer[INDEX(0, i)].s0123 = stateBuffer[INDEX(1, i)].s0123 = stateBuffer[INDEX(2, i)].s0123;
	stateBuffer[INDEX(size.x - 1, i)].s0123 = stateBuffer[INDEX(size.x - 2, i)].s0123 = stateBuffer[INDEX(size.x - 3, i)].s0123;
}

__kernel void stateBoundaryFreeFlowY(
	__global real8* stateBuffer,
	int2 size)
{
	int i = get_global_id(0);
	stateBuffer[INDEX(i, 0)].s0123 = stateBuffer[INDEX(i, 1)].s0123 = stateBuffer[INDEX(i, 2)].s0123;
	stateBuffer[INDEX(i, size.y - 1)].s0123 = stateBuffer[INDEX(i, size.y - 2)].s0123 = stateBuffer[INDEX(i, size.y - 3)].s0123;
}

__kernel void convertToTex(
	const __global real8* stateBuffer,
	const __global real* gravityPotentialBuffer,
	int2 size,
	__write_only image2d_t fluidTex,
	__read_only image1d_t gradientTex,
	int displayMethod,
	float displayScale)
{
	int2 i = (int2)(get_global_id(0), get_global_id(1));
	int index = INDEXV(i);
	
	real4 state = stateBuffer[index].s0123;
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
			real energyTotal = state.s3 / density;
			real energyKinetic = .5f * dot(state.yz, state.s12) / (density * density);
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

#if DIM == 1
	write_imagef(fluidTex, i, (float4)(value, 0.f, 0.f, 1.f));
#elif DIM == 2
	float4 color = read_imagef(gradientTex, CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_REPEAT | CLK_FILTER_LINEAR, value);
	write_imagef(fluidTex, i, color.bgra);
#endif
}

__kernel void poissonRelax(
	__global real* gravityPotentialBuffer,
	const __global real8* stateBuffer,
	int2 size,
	real2 dx,
	int2 repeat)
{
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
	
	real scale = M_PI * GRAVITY_CONSTANT * dx.x;
#if DIM > 1
	scale *= dx.y;
#endif

	gravityPotentialBuffer[index] = sum / (2.f * (float)DIM) + scale * stateBuffer[index].x;
}

__kernel void addGravity(
	__global real8* stateBuffer,
	const __global real* gravityPotentialBuffer,
	int2 size,
	real2 dx,
	const __global real* dtBuffer)
{
	real dt = dtBuffer[0];
	real2 dt_dx = dt / dx;

	int2 i = (int2)(get_global_id(0), get_global_id(1));
	if (i.x < 2 || i.y < 2 || i.x >= size.x - 2 || i.y >= size.y - 2) return;
	int index = i.x + size.x * i.y;

	real4 state = stateBuffer[index].s0123;
	real density = state.x;

	for (int side = 0; side < 2; ++side) {
		int2 iPrev = i;
		--iPrev[side];
		int indexPrev = iPrev.x + size.x * iPrev.y;
		
		int2 iNext = i;
		++iNext[side];
		int indexNext = iNext.x + size.x * iNext.y;	
	
		real gravityGrad = .5f * (gravityPotentialBuffer[indexNext] - gravityPotentialBuffer[indexPrev]);
		
		state[side+1] -= dt_dx[side] * density * gravityGrad;
		state.w -= dt * density * gravityGrad * state[side+1];
	}

	stateBuffer[index].s0123 = state;
}

__kernel void addDrop(
	__global real8* stateBuffer,
	int2 size,
	real2 xmin,
	real2 xmax,
	const __global real* dt,
	real2 pos,
	real2 sourceVelocity)
{
return;
	int2 i = (int2)(get_global_id(0), get_global_id(1));
	if (i.x >= size.x || i.y >= size.y) return;

	int index = i.x + size.x * i.y;
	real8 state = stateBuffer[index];

	const float dropRadius = .02f;
	const float densityMagnitude = 50.f;
	const float velocityMagnitude = 10000.f;
	const float energyInternalMagnitude = 0.f;
	
	real cellPosX = (real)i.x / (real)size.x * (xmax.x - xmin.x) + xmin.x;
	real cellPosY = (real)i.y / (real)size.y * (xmax.y - xmin.y) + xmin.y;
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

	stateBuffer[index].s0123 = (real4)(1.f, velocity.x, velocity.y, energyTotal) * density;
}

__kernel void addSource(
	__global real8* stateBuffer,
	int2 size,
	real2 xmin,
	real2 xmax,
	const __global real* dt)
{
return;
	int2 i = (int2)(get_global_id(0), get_global_id(1));
	if (i.x >= size.x || i.y >= size.y) return;

	const float radius = .05f;

	real cellPosX = (real)i.x / (real)size.x * (xmax.x - xmin.x) + xmin.x;
	real cellPosY = (real)i.y / (real)size.y * (xmax.y - xmin.y) + xmin.y;
	real2 cellPos = (real2)(cellPosX, cellPosY);
	real2 sourcePos = (real2)(xmin.x, .5f * (xmax.y + xmin.y));
	real2 dx = (cellPos - sourcePos) / radius;
	real rSq = dot(dx,dx);
	real falloff = exp(-rSq);

	int index = i.x + size.x * i.y;
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

	stateBuffer[index].s0123 = (real4)(1.f, velocity.x, velocity.y, energyTotal) * density;
}


