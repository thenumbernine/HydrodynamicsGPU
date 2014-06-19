#include "HydroGPU/Shared/Common2D.h"

real4 slopeLimiter(real4 r);

__kernel void calcCFL(
	__global real* cflBuffer,
	const __global real4* stateBuffer,
	const __global real* gravityPotentialBuffer,
	int2 size,
	real2 dx,
	real cfl)
{
	int2 i = (int2)(get_global_id(0), get_global_id(1));
	int index = INDEXV(i);
	if (i.x < 2 || i.x >= size.x - 2
#if DIM > 1
		|| i.y < 2 || i.y >= size.y - 2
#endif
	) {
		cflBuffer[index] = INFINITY;
		return;
	}

	real4 state = stateBuffer[index];
	real density = state.x;
	real2 velocity = state.yz / density;
	real specificTotalEnergy = state.w / density;
	real specificKineticEnergy = .5f * dot(velocity, velocity);
	real specificPotentialEnergy = gravityPotentialBuffer[index]; 
	real specificInternalEnergy = specificTotalEnergy - specificKineticEnergy - specificPotentialEnergy;
	real speedOfSound = sqrt(GAMMA * (GAMMA - 1.f) * specificInternalEnergy);
	real result = dx[0] / (speedOfSound + fabs(velocity[0]));
	for (int side = 0; side < DIM; ++side) {
		real dum = dx[side] / (speedOfSound + fabs(velocity[side]));
		result = min(result, dum);
	}
	cflBuffer[index] = cfl * result;
}

__kernel void calcInterfaceVelocity(
	__global real2* interfaceVelocityBuffer,
	const __global real4* stateBuffer,
	int2 size,
	real2 dx)
{
	int2 i = (int2)(get_global_id(0), get_global_id(1));
	if (i.x < 2 || i.x >= size.x - 1 
#if DIM > 1
		|| i.y < 2 || i.y >= size.y - 1
#endif
	) return;
	int index = INDEXV(i);
	
	real2 interfaceVelocity;
	for (int side = 0; side < DIM; ++side) {
		int2 iPrev = i;
		--iPrev[side];
		int indexPrev = INDEXV(iPrev);
		
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
	if (i.x < 2 || i.x >= size.x - 1 
#if DIM > 1
		|| i.y < 2 || i.y >= size.y - 1
#endif
	) return;
	int index = INDEXV(i);
	
	for (int side = 0; side < DIM; ++side) {	
		int2 iPrev = i;
		--iPrev[side];
		int indexPrev = INDEXV(iPrev);
	
		int2 iPrev2 = iPrev;
		--iPrev2[side];
		int indexPrev2 = INDEXV(iPrev2);
		
		int2 iNext = i;
		++iNext[side];
		int indexNext = INDEXV(iNext);

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
	if (i.x < 2 || i.x >= size.x - 2 
#if DIM > 1
		|| i.y < 2 || i.y >= size.y - 2
#endif
	) return;
	int index = INDEXV(i);
	
	for (int side = 0; side < DIM; ++side) {
		int2 iNext = i;
		++iNext[side];
		int indexNext = INDEXV(iNext);
		
		real4 fluxL = fluxBuffer[side + 2 * index];
		real4 fluxR = fluxBuffer[side + 2 * indexNext];
	
		real4 df = fluxR - fluxL;
		stateBuffer[index] -= df * dt_dx[side];
	}
}

__kernel void computePressure(
	__global real* pressureBuffer,
	const __global real4* stateBuffer,
	const __global real* gravityPotentialBuffer,
	int2 size)
{
	int2 i = (int2)(get_global_id(0), get_global_id(1));
	if (i.x < 1 || i.x >= size.x - 1 
#if DIM > 1
		|| i.y < 1 || i.y >= size.y - 1
#endif
	) return;
	int index = INDEXV(i);
	
	real4 state = stateBuffer[index];
	
	real density = state.x;
	real2 velocity = state.yz / density;
	real energyTotal = state.w / density;
	
	real energyKinetic = .5f * dot(velocity, velocity);
	real energyPotential = gravityPotentialBuffer[index];
	real energyInternal = energyTotal - energyKinetic - energyPotential;
	
	real pressure = (GAMMA - 1.f) * density * energyInternal;

	//von Neumann-Richtmyer artificial viscosity
	real deltaVelocitySq = 0.f;
	for (int side = 0; side < DIM; ++side) {
		int2 iPrev = i;
		--iPrev[side];
		int indexPrev = INDEXV(iPrev);
		
		int2 iNext = i;
		++iNext[side];
		int indexNext = INDEXV(iNext);

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
	if (i.x < 2 || i.x >= size.x - 2 
#if DIM > 1
		|| i.y < 2 || i.y >= size.y - 2
#endif
	) return;
	int index = INDEXV(i);

	for (int side = 0; side < DIM; ++side) {
		int2 iPrev = i;
		--iPrev[side];
		int indexPrev = INDEXV(iPrev);
		
		int2 iNext = i;
		++iNext[side];
		int indexNext = INDEXV(iNext);

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
	if (i.x < 2 || i.x >= size.x - 2
#if DIM > 1
		|| i.y < 2 || i.y >= size.y - 2
#endif
	) return;
	int index = INDEXV(i);

	for (int side = 0; side < DIM; ++side) {
		int2 iPrev = i;
		--iPrev[side];
		int indexPrev = INDEXV(iPrev);
		
		int2 iNext = i;
		++iNext[side];
		int indexNext = INDEXV(iNext);

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


