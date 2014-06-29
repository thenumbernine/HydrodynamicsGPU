#include "HydroGPU/Shared/Common.h"

__kernel void calcCFL(
	__global real* cflBuffer,
	const __global real8* stateBuffer,
	const __global real* gravityPotentialBuffer,
	real cfl)
{
	real4 dx = (real4)(DX, DY, DZ, 1.f);
	int4 i = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);
	int index = INDEXV(i);
	if (i.x < 2 || i.x >= SIZE_X - 2 
#if DIM > 1
		|| i.y < 2 || i.y >= SIZE_Y - 2 
#if DIM > 2
		|| i.z < 2 || i.z >= SIZE_Z - 2
#endif
#endif
	) {
		cflBuffer[index] = INFINITY;
		return;
	}

	real8 state = stateBuffer[index];
	real density = state.s0;
	real4 velocity = (real4)(state.s1, state.s2, state.s3, 0.f) / density;
	real specificTotalEnergy = state.s4 / density;
	real specificKineticEnergy = .5f * dot(velocity, velocity);
	real specificPotentialEnergy = gravityPotentialBuffer[index]; 
	real specificInternalEnergy = specificTotalEnergy - specificKineticEnergy - specificPotentialEnergy;
	real speedOfSound = sqrt(GAMMA * (GAMMA - 1.f) * specificInternalEnergy);
	real result = dx.s0 / (speedOfSound + fabs(velocity.s0));
	for (int side = 1; side < DIM; ++side) {
		real dum = dx[side] / (speedOfSound + fabs(velocity[side]));
		result = min(result, dum);
	}
	cflBuffer[index] = cfl * result;
}

__kernel void calcInterfaceVelocity(
	__global real* interfaceVelocityBuffer,
	const __global real8* stateBuffer)
{
	int4 i = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);
	if (i.x < 2 || i.x >= SIZE_X - 1 
#if DIM > 1
		|| i.y < 2 || i.y >= SIZE_Y - 1 
#if DIM > 2
		|| i.z < 2 || i.z >= SIZE_Z - 1
#endif
#endif
	) {
		return;
	}
	int index = INDEXV(i);
	
	for (int side = 0; side < DIM; ++side) {
		int4 iPrev = i;
		--iPrev[side];
		int indexPrev = INDEXV(iPrev);
		
		real8 stateL = stateBuffer[indexPrev];
		real densityL = stateL.s0;
		real velocityL = stateL[side+1] / densityL;
		
		real8 stateR = stateBuffer[index];
		real densityR = stateR.s0;
		real velocityR = stateR[side+1] / densityR;
		
		interfaceVelocityBuffer[side + DIM * index] = .5f * (velocityL + velocityR);
	}
}

__kernel void calcFlux(
	__global real8* fluxBuffer,
	const __global real8* stateBuffer,
	const __global real* interfaceVelocityBuffer,
	const __global real* dtBuffer)
{
	real4 dx = (real4)(DX, DY, DZ, 1.f);
	real dt = dtBuffer[0];
	real4 dt_dx = dt / dx;
	
	int4 i = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);
	if (i.x < 2 || i.x >= SIZE_X - 1 
#if DIM > 1
		|| i.y < 2 || i.y >= SIZE_Y - 1 
#if DIM > 2
		|| i.z < 2 || i.z >= SIZE_Z - 1
#endif
#endif
	) {
		return;
	}
	int index = INDEXV(i);
	
	for (int side = 0; side < DIM; ++side) {	
		int4 iPrev = i;
		iPrev[side] = iPrev[side]- 1;
		int indexPrev = INDEXV(iPrev);
	
		int4 iPrev2 = iPrev;
		--iPrev2[side];
		int indexPrev2 = INDEXV(iPrev2);
		
		int4 iNext = i;
		++iNext[side];
		int indexNext = INDEXV(iNext);

		int indexL1 = indexPrev;
		int indexR1 = index;
		int indexL2 = indexPrev2;
		int indexR2 = indexNext;

		real8 stateL = stateBuffer[indexL1];
		real8 stateR = stateBuffer[indexR1];
		real8 stateL2 = stateBuffer[indexL2];
		real8 stateR2 = stateBuffer[indexR2];

		real8 deltaStateL = stateL - stateL2;
		real8 deltaState = stateR - stateL;
		real8 deltaStateR = stateR2 - stateR;

		real interfaceVelocity = interfaceVelocityBuffer[side + DIM * index];
		real theta = step(0.f, interfaceVelocity);
	
		//this line crashes when compiling on my Intel HD4000 only for the 3D case
		//real8 stateSlopeRatio = mix(deltaStateR, deltaStateL, theta) / deltaState;
		//...but writing it out explicitly works fine
		real8 stateSlopeRatio;
		if (interfaceVelocity >= 0.f) {
			stateSlopeRatio = deltaStateL / deltaState;
		} else {
			stateSlopeRatio = deltaStateR / deltaState;
		}
		
		real8 phi = slopeLimiter(stateSlopeRatio);
		real8 delta = phi * deltaState;

		real8 flux = mix(stateR, stateL, theta) * interfaceVelocity;
		flux += delta * .5f * fabs(interfaceVelocity) * (1.f - fabs(interfaceVelocity * dt_dx[side])) / (float)DIM;
			
		fluxBuffer[side + DIM * index] = flux;
	}
}

__kernel void integrateFlux(
	__global real8* stateBuffer,
	const __global real8* fluxBuffer,
	const __global real* dtBuffer)
{
	real4 dx = (real4)(DX, DY, DZ, 1.f);
	real dt = dtBuffer[0];
	real4 dt_dx = dt / dx;
	
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
	
	for (int side = 0; side < DIM; ++side) {
		int4 iNext = i;
		++iNext[side];
		int indexNext = INDEXV(iNext);
		
		real8 fluxL = fluxBuffer[side + DIM * index];
		real8 fluxR = fluxBuffer[side + DIM * indexNext];
	
		real8 df = fluxR - fluxL;
		stateBuffer[index] -= df * dt_dx[side];
	}
}

__kernel void computePressure(
	__global real* pressureBuffer,
	const __global real8* stateBuffer,
	const __global real* gravityPotentialBuffer)
{
	int4 i = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);
	if (i.x < 1 || i.x >= SIZE_X - 1 
#if DIM > 1
		|| i.y < 1 || i.y >= SIZE_Y - 1 
#if DIM > 2
		|| i.z < 1 || i.z >= SIZE_Z - 1
#endif
#endif
	) {
		return;
	}
	int index = INDEXV(i);
	
	real8 state = stateBuffer[index];
	
	real density = state.s0;
	real4 velocity = (real4)(state.s1, state.s2, state.s3, 0.f) / density;
	real energyTotal = state.s4 / density;
	
	real energyKinetic = .5f * dot(velocity, velocity);
	real energyPotential = gravityPotentialBuffer[index];
	real energyInternal = energyTotal - energyKinetic - energyPotential;
	
	real pressure = (GAMMA - 1.f) * density * energyInternal;
	
	//von Neumann-Richtmyer artificial viscosity
	real deltaVelocitySq = 0.f;
	for (int side = 0; side < DIM; ++side) {
		int4 iPrev = i;
		--iPrev[side];
		int indexPrev = INDEXV(iPrev);
		
		int4 iNext = i;
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
	__global real8* stateBuffer,
	const __global real* pressureBuffer,
	const __global real* dtBuffer)
{
	real4 dx = (real4)(DX, DY, DZ, 1.f);
	real dt = dtBuffer[0];
	real4 dt_dx = dt / dx;
	
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

	for (int side = 0; side < DIM; ++side) {
		int4 iPrev = i;
		--iPrev[side];
		int indexPrev = INDEXV(iPrev);
		
		int4 iNext = i;
		++iNext[side];
		int indexNext = INDEXV(iNext);

		real pressureL = pressureBuffer[indexPrev];
		real pressureR = pressureBuffer[indexNext];

		real deltaPressure = .5f * (pressureR - pressureL);
		stateBuffer[index][side+1] -= deltaPressure * dt_dx[side];
	}
}

__kernel void diffuseWork(
	__global real8* stateBuffer,
	const __global real* pressureBuffer,
	const __global real* dtBuffer)
{
	real4 dx = (real4)(DX, DY, DZ, 1.f);
	real dt = dtBuffer[0];
	real4 dt_dx = dt / dx;
	
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

	for (int side = 0; side < DIM; ++side) {
		int4 iPrev = i;
		--iPrev[side];
		int indexPrev = INDEXV(iPrev);
		
		int4 iNext = i;
		++iNext[side];
		int indexNext = INDEXV(iNext);

		real8 stateL = stateBuffer[indexPrev];
		real8 stateR = stateBuffer[indexNext];

		real velocityL = stateL[side+1] / stateL.s0;
		real velocityR = stateR[side+1] / stateR.s0;
		
		real pressureL = pressureBuffer[indexPrev];
		real pressureR = pressureBuffer[indexNext];

		real deltaWork = .5f * (pressureR * velocityR - pressureL * velocityL);

		stateBuffer[index].s4 -= deltaWork * dt_dx[side];
	}
}

