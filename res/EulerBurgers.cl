#include "HydroGPU/Shared/Common.h"

__kernel void calcCFL(
	__global real* cflBuffer,
	const __global real* stateBuffer,
	const __global real* potentialBuffer,
	real cfl)
{
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

	real density = stateBuffer[STATE_DENSITY + NUM_STATES * index];
	real energyTotal = stateBuffer[STATE_ENERGY_TOTAL + NUM_STATES * index];
#if DIM == 1
	real velocity = stateBuffer[STATE_MOMENTUM_X + NUM_STATES * index] / density;
#elif DIM == 2
	real2 velocity = (real2)(stateBuffer[STATE_MOMENTUM_X + NUM_STATES * index], stateBuffer[STATE_MOMENTUM_Y + NUM_STATES * index]) / density;
#elif DIM == 3
	real4 velocity = (real4)(
		stateBuffer[STATE_MOMENTUM_X + NUM_STATES * index],
		stateBuffer[STATE_MOMENTUM_Y + NUM_STATES * index],
		stateBuffer[STATE_MOMENTUM_Z + NUM_STATES * index],
		0.f) / density;
#endif
	real specificEnergyTotal = energyTotal / density;
	real specificEnergyKinetic = .5f * dot(velocity, velocity);
	real specificEnergyPotential = potentialBuffer[index];
	real specificEnergyInternal = specificEnergyTotal - specificEnergyKinetic - specificEnergyPotential;

	real speedOfSound = sqrt(gamma * (gamma - 1.f) * specificEnergyInternal);
#if DIM == 1
	real result = dx.s0 / (speedOfSound + fabs(velocity));
#else
	real result = dx.s0 / (speedOfSound + fabs(velocity.s0));
	for (int side = 1; side < DIM; ++side) {
		real dum = dx[side] / (speedOfSound + fabs(velocity[side]));
		result = min(result, dum);
	}
#endif
	cflBuffer[index] = cfl * result;
}

__kernel void calcInterfaceVelocity(
	__global real* interfaceVelocityBuffer,
	const __global real* stateBuffer)
{
	int4 i = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);
	if (i.x < 2 || i.x >= SIZE_X - 1 
#if DIM > 1
		|| i.y < 2 || i.y >= SIZE_Y - 1 
#endif
#if DIM > 2
		|| i.z < 2 || i.z >= SIZE_Z - 1
#endif
	) {
		return;
	}
	int index = INDEXV(i);
	int indexR = index;

	for (int side = 0; side < DIM; ++side) {
		int indexL = index - stepsize[side];
		
		real densityL = stateBuffer[STATE_DENSITY + NUM_STATES * indexL];
		real velocityL = stateBuffer[side+STATE_MOMENTUM_X + NUM_STATES * indexL] / densityL;
		
		real densityR = stateBuffer[STATE_DENSITY + NUM_STATES * indexR];
		real velocityR = stateBuffer[side+STATE_MOMENTUM_X + NUM_STATES * indexR] / densityR;
		
		interfaceVelocityBuffer[side + DIM * index] = .5f * (velocityL + velocityR);
	}
}

__kernel void calcFlux(
	__global real* fluxBuffer,
	const __global real* stateBuffer,
	const __global real* interfaceVelocityBuffer,
	const __global real* dtBuffer)
{
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
	int indexR = index;

	for (int side = 0; side < DIM; ++side) {	
		int indexL = index - stepsize[side];
		int indexL2 = indexL - stepsize[side];
		int indexR2 = index + stepsize[side];

		real interfaceVelocity = interfaceVelocityBuffer[side + DIM * index];
		//real theta = step(0.f, interfaceVelocity);
	
		for (int j = 0; j < NUM_STATES; ++j) {
			real stateR2 = stateBuffer[j + NUM_STATES * indexR2];
			real stateR = stateBuffer[j + NUM_STATES * indexR];
			real stateL = stateBuffer[j + NUM_STATES * indexL];
			real stateL2 = stateBuffer[j + NUM_STATES * indexL2];
			
			real deltaStateL = stateL - stateL2;
			real deltaState = stateR - stateL;
			real deltaStateR = stateR2 - stateR;
			
			//3D case crashes?
			//real flux = mix(stateR, stateL, theta) * interfaceVelocity;

			//this line crashes when compiling on my Intel HD4000 only for the 3D case
			//real stateSlopeRatio = mix(deltaStateR, deltaStateL, theta) / deltaState;
			//...but writing it out explicitly works fine
			real stateSlopeRatio;
			real flux;
			if (interfaceVelocity >= 0.f) {
				stateSlopeRatio = deltaStateL / deltaState;
				flux = stateL * interfaceVelocity;
			} else {
				stateSlopeRatio = deltaStateR / deltaState;
				flux = stateR * interfaceVelocity;
			}

			//2nd order
			real phi = slopeLimiter(stateSlopeRatio);
			real delta = phi * deltaState;
			flux += delta * .5f * fabs(interfaceVelocity) * (1.f - fabs(interfaceVelocity * dt_dx[side])) / (float)DIM;
			
			fluxBuffer[j + NUM_STATES * (side + DIM * index)] = flux;
		}
	}
}

__kernel void calcFluxDeriv(
	__global real* derivBuffer,	//dstate/dt
	const __global real* fluxBuffer)
{
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
	
	__global real* deriv = derivBuffer + NUM_STATES * index;

	for (int side = 0; side < DIM; ++side) {
		int indexNext = index + stepsize[side];
		for (int j = 0; j < NUM_STATES; ++j) {
			real fluxL = fluxBuffer[j + NUM_STATES * (side + DIM * index)];
			real fluxR = fluxBuffer[j + NUM_STATES * (side + DIM * indexNext)];
			real deltaFlux = fluxR - fluxL;
			deriv[j] -= deltaFlux / dx[side];
		}
	}
}

__kernel void computePressure(
	__global real* pressureBuffer,
	const __global real* stateBuffer,
	const __global real* potentialBuffer)
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

	real density = stateBuffer[STATE_DENSITY + NUM_STATES * index];
	real energyTotal = stateBuffer[STATE_ENERGY_TOTAL + NUM_STATES * index];
#if DIM == 1
	real velocity = stateBuffer[STATE_MOMENTUM_X + NUM_STATES * index] / density;
#elif DIM == 2
	real2 velocity = (real2)(stateBuffer[STATE_MOMENTUM_X + NUM_STATES * index], stateBuffer[STATE_MOMENTUM_Y + NUM_STATES * index]) / density;
#elif DIM == 3
	real4 velocity = (real4)(
		stateBuffer[STATE_MOMENTUM_X + NUM_STATES * index],
		stateBuffer[STATE_MOMENTUM_Y + NUM_STATES * index],
		stateBuffer[STATE_MOMENTUM_Z + NUM_STATES * index],
		0.f) / density;
#endif
	real specificEnergyTotal = energyTotal / density;
	real specificEnergyKinetic = .5f * dot(velocity, velocity);
	real specificEnergyPotential = potentialBuffer[index];
	real specificEnergyInternal = specificEnergyTotal - specificEnergyKinetic - specificEnergyPotential;
	real pressure = (gamma - 1.f) * density * specificEnergyInternal;
	//von Neumann-Richtmyer artificial viscosity
	real deltaVelocitySq = 0.f;
	for (int side = 0; side < DIM; ++side) {
		int indexL = index - stepsize[side];
		int indexR = index + stepsize[side];	

		real velocityL = stateBuffer[side+STATE_MOMENTUM_X + NUM_STATES * indexL];
		real velocityR = stateBuffer[side+STATE_MOMENTUM_X + NUM_STATES * indexR];
		const float ZETA = 2.f;
		real deltaVelocity = ZETA * .5f * (velocityR - velocityL);
		deltaVelocitySq += deltaVelocity * deltaVelocity; 
	}
	pressure += deltaVelocitySq * density;
	pressureBuffer[index] = pressure;
}

__kernel void diffuseMomentum(
	__global real* derivBuffer,
	const __global real* pressureBuffer)
{
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

	__global real* deriv = derivBuffer + NUM_STATES * index;

	for (int side = 0; side < DIM; ++side) {
		int indexL = index - stepsize[side];
		int indexR = index + stepsize[side];	

		real pressureL = pressureBuffer[indexL];
		real pressureR = pressureBuffer[indexR];

		real deltaPressure = .5f * (pressureR - pressureL);
		deriv[side + STATE_MOMENTUM_X] -= deltaPressure / dx[side];
	}
}

__kernel void diffuseWork(
	__global real* derivBuffer,
	const __global real* stateBuffer,
	const __global real* pressureBuffer)
{
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

	__global real* deriv = derivBuffer + NUM_STATES * index;

	for (int side = 0; side < DIM; ++side) {
		int indexL = index - stepsize[side];
		int indexR = index + stepsize[side];

		real velocityL = stateBuffer[side+STATE_MOMENTUM_X + NUM_STATES * indexL] / stateBuffer[STATE_DENSITY + NUM_STATES * indexL];
		real velocityR = stateBuffer[side+STATE_MOMENTUM_X + NUM_STATES * indexR] / stateBuffer[STATE_DENSITY + NUM_STATES * indexR];
		
		real pressureL = pressureBuffer[indexL];
		real pressureR = pressureBuffer[indexR];

		real deltaWork = .5f * (pressureR * velocityR - pressureL * velocityL);

		deriv[STATE_ENERGY_TOTAL] -= deltaWork / dx[side];
	}
}

