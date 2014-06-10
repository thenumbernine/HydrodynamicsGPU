#include "HydroGPU/Shared/Types.h"

real4 slopeLimiter(real4 r);

__kernel void calcCFL(
	__global real* cflBuffer,
	const __global real4* stateBuffer,
	int2 size,
	real2 dx,
	real cfl)
{
	int2 i = (int2)(get_global_id(0), get_global_id(1));
	if (i.x >= size.x || i.y >= size.y) return;
	int index = i.x + size.x * i.y;

	real4 state = stateBuffer[index];
	real density = state.x;
	real2 velocity = state.yz / density;
	real specificTotalEnergy = state.w / density;
	real specificKineticEnergy = .5f * dot(velocity, velocity);
	real specificInternalEnergy = specificTotalEnergy - specificKineticEnergy;
	real speedOfSound = sqrt(GAMMA * (GAMMA - 1.f) * specificInternalEnergy);
	real dumx = dx.x / (speedOfSound + fabs(velocity.x));
	real dumy = dx.y / (speedOfSound + fabs(velocity.y));
	cflBuffer[index] = cfl * min(dumx, dumy);
}

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

__kernel void calcInterfaceVelocity(
	__global real2* interfaceVelocityBuffer,
	const __global real4* stateBuffer,
	int2 size,
	real2 dx)
{
	int2 i = (int2)(get_global_id(0), get_global_id(1));
	if (i.x >= size.x || i.y >= size.y) return;
	int index = i.x + size.x * i.y;
	
	real2 interfaceVelocity;
	for (int side = 0; side < 2; ++side) {
		int2 iPrev = i;
		iPrev[side] = (iPrev[side] + size[side] - 1) % size[side];
		int indexPrev = iPrev.x + size.x * iPrev.y;
		
		real4 stateL = stateBuffer[indexPrev];
		real densityL = stateL.x;
		real velocityL = stateL[side+1] / densityL;
		
		real4 stateR = stateBuffer[index];
		real densityR = stateR.x;
		real velocityR = stateR[side+1] / densityR;
		
		interfaceVelocity[side] = .5f * (velocityL + velocityR);
	}
	interfaceVelocityBuffer[index] = interfaceVelocity;
}

real4 slopeLimiter(real4 r) {
	//donor cell
	//return (real4)(0.f, 0.f, 0.f, 0.f);
	//Lax-Wendroff
	//return (real4)(1.f, 1.f, 1.f, 1.f);
	//superbee
	return max(0.f, max(min(1.f, 2.f * r), min(2.f, r)));
}

__kernel void calcFlux(
	__global real4* fluxBuffer,
	const __global real4* stateBuffer,
	const __global real2* interfaceVelocityBuffer,
	int2 size,
	real2 dx,
	const __global real* dtBuffer)
{
	real dt = dtBuffer[0];
	real2 dt_dx = dt / dx;
	
	int2 i = (int2)(get_global_id(0), get_global_id(1));
	if (i.x >= size.x || i.y >= size.y) return;
	int index = i.x + size.x * i.y;
	
	for (int side = 0; side < 2; ++side) {	
		int2 iPrev = i;
		iPrev[side] = (iPrev[side] + size[side] - 1) % size[side];
		int indexPrev = iPrev.x + size.x * iPrev.y;
	
		int2 iPrev2 = iPrev;
		iPrev2[side] = (iPrev2[side] + size[side] - 1) % size[side];
		int indexPrev2 = iPrev2.x + size.x * iPrev2.y;
		
		int2 iNext = i;
		iNext[side] = (iNext[side] + 1) % size[side];
		int indexNext = iNext.x + size.x * iNext.y;

		int indexL2 = indexPrev2;
		int indexL1 = indexPrev;
		int indexR1 = index;
		int indexR2 = indexNext;

		real4 stateL2 = stateBuffer[indexL2];
		real4 stateL = stateBuffer[indexL1];
		real4 stateR = stateBuffer[indexR1];
		real4 stateR2 = stateBuffer[indexR2];

		real4 deltaStateL = stateL - stateL2;
		real4 deltaState = stateR - stateL;
		real4 deltaStateR = stateR2 - stateR;

		real interfaceVelocity = interfaceVelocityBuffer[index][side];
		real theta = step(0.f, interfaceVelocity);
		
		real4 stateSlopeRatio = mix(deltaStateR, deltaStateL, theta) / deltaState;
		real4 phi = slopeLimiter(stateSlopeRatio);
		real4 delta = phi * deltaState;

		real4 flux = mix(stateR, stateL, theta) * interfaceVelocity
			+ delta * .5f * (.5f * fabs(interfaceVelocity) * (1.f - fabs(interfaceVelocity * dt_dx[side])));

		fluxBuffer[side + 2 * index] = flux;
	}
}

__kernel void integrateFlux(
	__global real4* stateBuffer,
	const __global real4* fluxBuffer,
	int2 size,
	real2 dx,
	const __global real* dtBuffer)
{
	real2 dt_dx = dtBuffer[0] / dx;
	
	int2 i = (int2)(get_global_id(0), get_global_id(1));
	if (i.x >= size.x || i.y >= size.y) return;

	int index = i.x + size.x * i.y;
	
	for (int side = 0; side < 2; ++side) {
		int2 iNext = i;
		iNext[side] = (iNext[side] + 1) % size[side];
		int indexNext = iNext.x + size.x * iNext.y;
		
		real4 fluxL = fluxBuffer[side + 2 * index];
		real4 fluxR = fluxBuffer[side + 2 * indexNext];
	
		real4 df = fluxR - fluxL;
		stateBuffer[index] -= df * dt_dx[side];
	}
}

__kernel void computePressure(
	__global real* pressureBuffer,
	const __global real4* stateBuffer,
	int2 size)
{
	int2 i = (int2)(get_global_id(0), get_global_id(1));
	if (i.x >= size.x || i.y >= size.y) return;
	int index = i.x + size.x * i.y;
	
	real4 state = stateBuffer[index];
	
	real density = state.x;
	real2 velocity = state.yz / density;
	real energyTotal = state.w / density;
	
	real energyKinetic = .5f * dot(velocity, velocity);
	real energyInternal = energyTotal - energyKinetic;
	
	real pressure = (GAMMA - 1.f) * density * energyInternal;

	//von Neumann-Richtmyer artiﬁcial viscosity
	real deltaVelocitySq = 0.f;
	for (int side = 0; side < 2; ++side) {
		int2 iPrev = i;
		iPrev[side] = (iPrev[side] + size[side] - 1) % size[side];
		int indexPrev = iPrev.x + size.x * iPrev.y;
		
		int2 iNext = i;
		iNext[side] = (iNext[side] + 1) % size[side];
		int indexNext = iNext.x + size.x * iNext.y;

		real velocityL = stateBuffer[indexPrev][side+1];
		real velocityR = stateBuffer[indexNext][side+1];
		const float ZETA = 2.f;
		real deltaVelocity = ZETA * .5f * (velocityR - velocityL);
		deltaVelocitySq += deltaVelocity * deltaVelocity; 
	}
	pressure += deltaVelocitySq * density;

	pressureBuffer[index] = pressure;
}

__kernel void diffuseMomentum(
	__global real4* stateBuffer,
	const __global real* pressureBuffer,
	int2 size,
	real2 dx,
	const __global real* dtBuffer)
{
	real2 dt_dx = dtBuffer[0] / dx;
	
	int2 i = (int2)(get_global_id(0), get_global_id(1));
	if (i.x >= size.x || i.y >= size.y) return;
	int index = i.x + size.x * i.y;

	for (int side = 0; side < 2; ++side) {
		int2 iPrev = i;
		iPrev[side] = (iPrev[side] + size[side] - 1) % size[side];
		int indexPrev = iPrev.x + size.x * iPrev.y;
		
		int2 iNext = i;
		iNext[side] = (iNext[side] + 1) % size[side];
		int indexNext = iNext.x + size.x * iNext.y;

		real pressureL = pressureBuffer[indexPrev];
		real pressureR = pressureBuffer[indexNext];

		real deltaPressure = .5f * (pressureR - pressureL);
		stateBuffer[index][side+1] -= deltaPressure * dt_dx[side];
	}
}

__kernel void diffuseWork(
	__global real4* stateBuffer,
	const __global real* pressureBuffer,
	int2 size,
	real2 dx,
	const __global real* dtBuffer)
{
	real2 dt_dx = dtBuffer[0] / dx;
	
	int2 i = (int2)(get_global_id(0), get_global_id(1));
	if (i.x >= size.x || i.y >= size.y) return;
	int index = i.x + size.x * i.y;

	for (int side = 0; side < 2; ++side) {
		int2 iPrev = i;
		iPrev[side] = (iPrev[side] + size[side] - 1) % size[side];
		int indexPrev = iPrev.x + size.x * iPrev.y;
		
		int2 iNext = i;
		iNext[side] = (iNext[side] + 1) % size[side];
		int indexNext = iNext.x + size.x * iNext.y;

		real4 stateL = stateBuffer[indexPrev];
		real4 stateR = stateBuffer[indexNext];

		real velocityL = stateL[side+1] / stateL.x;
		real velocityR = stateR[side+1] / stateR.x;
		
		real pressureL = pressureBuffer[indexPrev];
		real pressureR = pressureBuffer[indexNext];

		real deltaWork = .5f * (pressureR * velocityR - pressureL * velocityL);

		stateBuffer[index].w -= deltaWork * dt_dx[side];
	}
}

__kernel void convertToTex(
	__global real4* stateBuffer,
	int2 size,
	__write_only image2d_t fluidTex,
	__read_only image2d_t gradientTex)
{
	int2 i = (int2)(get_global_id(0), get_global_id(1));
	if (i.x >= size.x || i.y >= size.y) return;
	int index = i.x + size.x * i.y;
	
	real4 state = stateBuffer[index];
	real value = state.x;	//density
	//real value = length(state.yz) / state.x;	//velocity
	//real value = (GAMMA - 1.f) * state.x * state.w;	//pressure

	float4 color = read_imagef(gradientTex, 
		CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_REPEAT | CLK_FILTER_LINEAR,
		(float2)(value * 2.f, .5f));
	write_imagef(fluidTex, i, color.bgra);
}

__kernel void addDrop(
	__global real4* stateBuffer,
	int2 size,
	real2 xmin,
	real2 xmax,
	real2 pos,
	real2 sourceVelocity)
{
	int2 i = (int2)(get_global_id(0), get_global_id(1));
	if (i.x >= size.x || i.y >= size.y) return;

	int index = i.x + size.x * i.y;
	real4 state = stateBuffer[index];

	float dropRadius = .02f;
	float densityMagnitude = .05f;
	float velocityMagnitude = 1.f;
	float energyThermalMagnitude = 0.f;

	real2 x = (real2)(i.x, i.y) / (real2)(size.x, size.y) * (xmax - xmin) + xmin;
	real2 dx = (x - pos) / dropRadius;
	float rSq = dot(dx, dx);
	float falloff = exp(-rSq);

	real density = state.x;
	real2 velocity = state.yz / density;
	real energyTotal = state.w / density;
	real energyKinetic = .5f * dot(velocity, velocity);
	real energyThermal = energyTotal - energyKinetic;

	density += densityMagnitude * falloff;
	velocity += sourceVelocity * (falloff * velocityMagnitude);
	energyThermal += energyThermalMagnitude * falloff;

	energyKinetic = .5f * dot(velocity, velocity);
	energyTotal = energyThermal + energyKinetic;

	stateBuffer[index] = (real4)(1.f, velocity.x, velocity.y, energyTotal) * density;
}

__kernel void addSource(
	__global real4* stateBuffer,
	int2 size,
	real2 xmin,
	real2 xmax,
	const __global real* dtBuffer)
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
	velocity.x += velocityMagnitude * falloff * dtBuffer[0];

	energyKinetic = .5f * dot(velocity, velocity);
	energyTotal = energyInternal + energyKinetic;

	stateBuffer[index] = (real4)(1.f, velocity.x, velocity.y, energyTotal) * density;
}

