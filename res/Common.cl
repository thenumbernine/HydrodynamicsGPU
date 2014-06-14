#include "HydroGPU/Shared/Common.h"

//periodic

__kernel void stateBoundaryPeriodicHorizontal(
	__global real4* stateBuffer,	//ghost
	int2 size)
{
	int i = get_global_id(0);
	if (i >= size.x) return;
	stateBuffer[(i+2) + (size.x+4)*0] = stateBuffer[(i+2) + (size.x+4)*size.y];
	stateBuffer[(i+2) + (size.x+4)*1] = stateBuffer[(i+2) + (size.x+4)*(size.y+1)];
	stateBuffer[(i+2) + (size.x+4)*(size.y+3)] = stateBuffer[(i+2) + (size.x+4)*3];
	stateBuffer[(i+2) + (size.x+4)*(size.y+2)] = stateBuffer[(i+2) + (size.x+4)*2];
}

__kernel void stateBoundaryPeriodicVertical(
	__global real4* stateBuffer,
	int2 size)
{
	int i = get_global_id(0);
	if (i >= size.y) return;
	stateBuffer[0 + (size.x+4)*(i+2)] = stateBuffer[size.x+0 + (size.x+4)*(i+2)];
	stateBuffer[1 + (size.x+4)*(i+2)] = stateBuffer[size.x+1 + (size.x+4)*(i+2)];
	stateBuffer[size.x+3 + (size.x+4)*(i+2)] = stateBuffer[3 + (size.x+4)*(i+2)];
	stateBuffer[size.x+2 + (size.x+4)*(i+2)] = stateBuffer[2 + (size.x+4)*(i+2)];
}

//mirror

__kernel void stateBoundaryMirrorHorizontal(
	__global real4* stateBuffer,	//ghost
	int2 size)
{
	int i = get_global_id(0);
	if (i >= size.x) return;
	stateBuffer[INDEXGHOST(i, -2)].xw =  stateBuffer[INDEXGHOST(i, 1)].xw;
	stateBuffer[INDEXGHOST(i, -2)].yz = -stateBuffer[INDEXGHOST(i, 1)].yz;
	stateBuffer[INDEXGHOST(i, -1)].xw =  stateBuffer[INDEXGHOST(i, 0)].xw;
	stateBuffer[INDEXGHOST(i, -1)].yz = -stateBuffer[INDEXGHOST(i, 0)].yz;
	stateBuffer[INDEXGHOST(i, size.y + 1)].xw =  stateBuffer[INDEXGHOST(i, size.y - 2)].xw;
	stateBuffer[INDEXGHOST(i, size.y + 1)].yz = -stateBuffer[INDEXGHOST(i, size.y - 2)].yz;
	stateBuffer[INDEXGHOST(i, size.y + 0)].xw =  stateBuffer[INDEXGHOST(i, size.y - 1)].xw;
	stateBuffer[INDEXGHOST(i, size.y + 0)].yz = -stateBuffer[INDEXGHOST(i, size.y - 1)].yz;
}

__kernel void stateBoundaryMirrorVertical(
	__global real4* stateBuffer,
	int2 size)
{
	int i = get_global_id(0);
	if (i >= size.y) return;
	stateBuffer[INDEXGHOST(-2, i)].xw =  stateBuffer[INDEXGHOST(1, i)].xw;
	stateBuffer[INDEXGHOST(-2, i)].yz = -stateBuffer[INDEXGHOST(1, i)].yz;
	stateBuffer[INDEXGHOST(-1, i)].xw =  stateBuffer[INDEXGHOST(0, i)].xw;
	stateBuffer[INDEXGHOST(-1, i)].yz = -stateBuffer[INDEXGHOST(0, i)].yz;
	stateBuffer[INDEXGHOST(size.x + 1, i)].xw =  stateBuffer[INDEXGHOST(size.x - 2, i)].xw;
	stateBuffer[INDEXGHOST(size.x + 1, i)].yz = -stateBuffer[INDEXGHOST(size.x - 2, i)].yz;
	stateBuffer[INDEXGHOST(size.x + 0, i)].xw =  stateBuffer[INDEXGHOST(size.x - 1, i)].xw;
	stateBuffer[INDEXGHOST(size.x + 0, i)].yz = -stateBuffer[INDEXGHOST(size.x - 1, i)].yz;	
}

//freeflow

__kernel void stateBoundaryFreeFlowHorizontal(
	__global real4* stateBuffer,	//ghost
	int2 size)
{
	int i = get_global_id(0);
	if (i >= size.x) return;	
	stateBuffer[INDEXGHOST(i, -2)] = stateBuffer[INDEXGHOST(i, -1)] = stateBuffer[INDEXGHOST(i, 0)];
	stateBuffer[INDEXGHOST(i, size.y + 1)] = stateBuffer[INDEXGHOST(i, size.y)] = stateBuffer[INDEXGHOST(i, size.y - 1)];
}


__kernel void stateBoundaryFreeFlowVertical(
	__global real4* stateBuffer,
	int2 size)
{
	int i = get_global_id(0);
	if (i >= size.y) return;
	stateBuffer[INDEXGHOST(-2, i)] = stateBuffer[INDEXGHOST(-1, i)] = stateBuffer[INDEXGHOST(0, i)];
	stateBuffer[INDEXGHOST(size.x + 1, i)] = stateBuffer[INDEXGHOST(size.x, i)] = stateBuffer[INDEXGHOST(size.x - 1, i)];
}

//convert to tex

__kernel void convertToTex(
	const __global real4* stateBuffer,	//ghost
	const __global real* gravityPotentialBuffer,	//ghost
	int2 size,
	__write_only image2d_t fluidTex,
	__read_only image2d_t gradientTex,
	int displayMethod,
	float displayScale)
{
	int2 i = (int2)(get_global_id(0), get_global_id(1));
	if (i.x >= size.x || i.y >= size.y) return;
	int indexWGhost = INDEXGHOSTV(i);
	
	real4 state = stateBuffer[indexWGhost];

	real value;
	switch (displayMethod) {
	case DISPLAY_DENSITY:	//density
		value = state.x;
		break;
	case DISPLAY_VELOCITY:	//velocity
		value = length(state.yz) / state.x;
		break;
	case DISPLAY_PRESSURE:	//pressure
		{
			real density = state.x;
			real energyTotal = state.w / density;
			real energyKinetic = .5f * dot(state.yz, state.yz) / (density * density);
			real energyPotential = gravityPotentialBuffer[indexWGhost];
			real energyInternal = energyTotal - energyKinetic - energyPotential;
			value = (GAMMA - 1.f) * energyInternal * density;
		}
		break;
	case DISPLAY_GRAVITY_POTENTIAL:
		value = gravityPotentialBuffer[indexWGhost];
		break;
	default:
		value = .5f;
		break;
	}
	value *= displayScale;

	float4 color = read_imagef(gradientTex, 
		CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_REPEAT | CLK_FILTER_LINEAR,
		(float2)(value, .5f));
	write_imagef(fluidTex, i, color.bgra);
}

__kernel void addDrop(
	__global real4* stateBuffer,
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
	real4 state = stateBuffer[index];

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

	stateBuffer[index] = (real4)(1.f, velocity.x, velocity.y, energyTotal) * density;
}

__kernel void addSource(
	__global real4* stateBuffer,
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
	real4 state = stateBuffer[index];
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
}

