#include "HydroGPU/Shared/Common.h"

__kernel void applyBoundaryHorizontal(
	__global real4* stateBuffer,
	int2 size,
	int boundaryMethod)
{
	int i = get_global_id(0);
	if (i >= size.x) return;
	
	switch (boundaryMethod) {
	case BOUNDARY_REPEAT:
#if 0
		stateBuffer[i + size.x * 0] = stateBuffer[i + size.x * (size.y - 4)];
		stateBuffer[i + size.x * 1] = stateBuffer[i + size.x * (size.y - 3)];
		
		stateBuffer[i + size.x * (size.y - 1)] = stateBuffer[i + size.x * 3];
		stateBuffer[i + size.x * (size.y - 2)] = stateBuffer[i + size.x * 2];
#endif
		break;
	case BOUNDARY_MIRROR:
		stateBuffer[i + size.x * 0].xw = stateBuffer[i + size.x * 3].xw;
		stateBuffer[i + size.x * 0].yz = -stateBuffer[i + size.x * 3].yz;
		stateBuffer[i + size.x * 1].xw = stateBuffer[i + size.x * 2].xw;
		stateBuffer[i + size.x * 1].yz = -stateBuffer[i + size.x * 2].yz;
		
		stateBuffer[i + size.x * (size.y - 1)].xw = stateBuffer[i + size.x * (size.y - 4)].xw;
		stateBuffer[i + size.x * (size.y - 1)].yz = -stateBuffer[i + size.x * (size.y - 4)].yz;
		stateBuffer[i + size.x * (size.y - 2)].xw = stateBuffer[i + size.x * (size.y - 3)].xw;
		stateBuffer[i + size.x * (size.y - 2)].yz = -stateBuffer[i + size.x * (size.y - 3)].yz;
		break;
	case BOUNDARY_FREEFLOW:
#if 0
		stateBuffer[i] = stateBuffer[i + size.x] = stateBuffer[i + size.x * 2];
		stateBuffer[i + size.x * (size.y - 1)] = stateBuffer[i + size.x * (size.y - 2)] = stateBuffer[i + size.x * (size.y - 3)];
#endif
		break;
	}
}

__kernel void applyBoundaryVertical(
	__global real4* stateBuffer,
	int2 size,
	int boundaryMethod)
{
	int i = get_global_id(0);
	if (i >= size.y) return;

	switch (boundaryMethod) {
	case BOUNDARY_REPEAT:
#if 0
		stateBuffer[0 + size.x * i] = stateBuffer[(size.x - 4) + size.x * i];
		stateBuffer[1 + size.x * i] = stateBuffer[(size.x - 3) + size.x * i];
		
		stateBuffer[(size.x - 1) + size.x * i] = stateBuffer[3 + size.x * i];
		stateBuffer[(size.x - 2) + size.x * i] = stateBuffer[2 + size.x * i];
#endif
		break;
	case BOUNDARY_MIRROR:
		stateBuffer[0 + size.x * i].xw = stateBuffer[3 + size.x * i].xw;
		stateBuffer[0 + size.x * i].yz = -stateBuffer[3 + size.x * i].yz;
		stateBuffer[1 + size.x * i].xw = stateBuffer[2 + size.x * i].xw;
		stateBuffer[1 + size.x * i].yz = -stateBuffer[2 + size.x * i].yz;
		
		stateBuffer[(size.x - 1) + size.x * i].xw = stateBuffer[(size.x - 4) + size.x * i].xw;
		stateBuffer[(size.x - 1) + size.x * i].yz = -stateBuffer[(size.x - 4) + size.x * i].yz;
		stateBuffer[(size.x - 2) + size.x * i].xw = stateBuffer[(size.x - 3) + size.x * i].xw;
		stateBuffer[(size.x - 2) + size.x * i].yz = -stateBuffer[(size.x - 3) + size.x * i].yz;	
		break;
	case BOUNDARY_FREEFLOW:
#if 0
		stateBuffer[size.x * i] = stateBuffer[1 + size.x * i] = stateBuffer[2 + size.x * i];
		stateBuffer[(size.x - 1) + size.x * i] = stateBuffer[(size.x - 2) + size.x * i] = stateBuffer[(size.x - 3) + size.x * i];
#endif
		break;
	}
}
__kernel void convertToTex(
	const __global real4* stateBuffer,
	const __global real* gravityPotentialBuffer,
	int2 size,
	__write_only image2d_t fluidTex,
	__read_only image2d_t gradientTex,
	int displayMethod,
	float displayScale)
{
	int2 i = (int2)(get_global_id(0), get_global_id(1));
	if (i.x >= size.x || i.y >= size.y) return;
	
	int index = i.x + size.x * i.y;
	real4 state = stateBuffer[index];

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

