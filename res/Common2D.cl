#include "HydroGPU/Shared/Common2D.h"

real4 mirrorStateX(real4 state);
real4 mirrorStateY(real4 state);
real4 freeflowStateX(real4 state);
real4 freeflowStateY(real4 state);

//periodic

__kernel void stateBoundaryPeriodicHorizontal(
	__global real4* stateBuffer,	//ghost
	int2 size)
{
	int i = get_global_id(0);
	stateBuffer[i + size.x * 0] = stateBuffer[i + size.x * (size.y - 4)];
	stateBuffer[i + size.x * 1] = stateBuffer[i + size.x * (size.y - 3)];
	stateBuffer[i + size.x * (size.y - 2)] = stateBuffer[i + size.x * 2];
	stateBuffer[i + size.x * (size.y - 1)] = stateBuffer[i + size.x * 3];
}

__kernel void stateBoundaryPeriodicVertical(
	__global real4* stateBuffer,
	int2 size)
{
	int i = get_global_id(0);
	stateBuffer[0 + size.x * i] = stateBuffer[size.x - 4 + size.x * i];
	stateBuffer[1 + size.x * i] = stateBuffer[size.x - 3 + size.x * i];
	stateBuffer[size.x - 2 + size.x * i] = stateBuffer[2 + size.x * i];
	stateBuffer[size.x - 1 + size.x * i] = stateBuffer[3 + size.x * i];
}

//mirror

real4 mirrorStateX(real4 state) { return (real4)(state.x, -state.y, state.z, state.w); }
real4 mirrorStateY(real4 state) { return (real4)(state.x, state.y, -state.z, state.w); }

__kernel void stateBoundaryMirrorHorizontal(
	__global real4* stateBuffer,	//ghost
	int2 size)
{
	int i = get_global_id(0);
	stateBuffer[i + size.x * 0] = mirrorStateY(stateBuffer[i + size.x * 3]);
	stateBuffer[i + size.x * 1] = mirrorStateY(stateBuffer[i + size.x * 2]);
	stateBuffer[i + size.x * (size.y - 1)] = mirrorStateY(stateBuffer[i + size.x * (size.y - 4)]);
	stateBuffer[i + size.x * (size.y - 2)] = mirrorStateY(stateBuffer[i + size.x * (size.y - 3)]);
}

__kernel void stateBoundaryMirrorVertical(
	__global real4* stateBuffer,
	int2 size)
{
	int i = get_global_id(0);
	stateBuffer[0 + size.x * i] = mirrorStateX(stateBuffer[3 + size.x * i]);
	stateBuffer[1 + size.x * i] = mirrorStateX(stateBuffer[2 + size.x * i]);
	stateBuffer[size.x - 1 + size.x * i] = mirrorStateX(stateBuffer[size.x - 4 + size.x * i]);
	stateBuffer[size.x - 2 + size.x * i] = mirrorStateX(stateBuffer[size.x - 3 + size.x * i]);	
}

//freeflow

__kernel void stateBoundaryFreeFlowHorizontal(
	__global real4* stateBuffer,	//ghost
	int2 size)
{
	int i = get_global_id(0);
	stateBuffer[i + size.x * 0] = stateBuffer[i + size.x * 1] = (stateBuffer[i + size.x * 2]);
	stateBuffer[i + size.x * (size.y - 1)] = stateBuffer[i + size.x * (size.y - 2)] = (stateBuffer[i + size.x * (size.y - 3)]);
}


__kernel void stateBoundaryFreeFlowVertical(
	__global real4* stateBuffer,
	int2 size)
{
	int i = get_global_id(0);
	stateBuffer[0 + size.x * i] = stateBuffer[1 + size.x * i] = (stateBuffer[2 + size.x * i]);
	stateBuffer[size.x - 1 + size.x * i] = stateBuffer[size.x - 2 + size.x * i] = (stateBuffer[size.x - 3 + size.x * i]);
}

__kernel void convertToTex(
	const __global real4* stateBuffer,
	const __global real* gravityPotentialBuffer,
	int2 size,
	__write_only image2d_t fluidTex,
	__read_only image1d_t gradientTex,
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

	float4 color = read_imagef(gradientTex, CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_REPEAT | CLK_FILTER_LINEAR, value);
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

__kernel void poissonRelax(
	__global real* gravityPotentialBuffer,
	const __global real4* stateBuffer,
	int2 size,
	real2 dx,
	char repeat)
{
	int2 i = (int2)(get_global_id(0), get_global_id(1));
	int index = i.x + size.x * i.y;

	int ixp, ixn, iyp, iyn;
	if (repeat) {
		ixp = (i.x + size.x - 1) % size.x;
		ixn = (i.x + 1) % size.x;
		iyp = (i.y + size.y - 1) % size.y;
		iyn = (i.y + 1) % size.y;
	} else {
		ixp = max(i.x - 1, 0);
		ixn = min(i.x + 1, size.x - 1);
		iyp = max(i.y - 1, 0);
		iyn = min(i.y + 1, size.y - 1);
	}

	int xp = ixp + size.x * i.y;
	int xn = ixn + size.x * i.y;
	int yp = i.x + size.x * iyp;
	int yn = i.x + size.x * iyn;

	real sum = gravityPotentialBuffer[xp] 
				+ gravityPotentialBuffer[xn] 
				+ gravityPotentialBuffer[yp] 
				+ gravityPotentialBuffer[yn];
	
#define M_PI 3.141592653589793115997963468544185161590576171875f
#define GRAVITY_CONSTANT 1.f		//6.67384e-11 m^3 / (kg s^2)
	real scale = M_PI * GRAVITY_CONSTANT * dx.x * dx.y;

	gravityPotentialBuffer[index] = .25f * sum + scale * stateBuffer[index].x;
}

__kernel void addGravity(
	__global real4* stateBuffer,
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

	real4 state = stateBuffer[index];
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

	stateBuffer[index] = state;
}


