#include "HydroGPU/Shared/Common.h"
#include "HydroGPU/Euler.h"

/*
Euler Burgers:
 
  d [ rho ]    d    [ rho ]     d [ 0 ]
  - [rho v] +  - (v [rho v]) +  - [ P ] = 0
 dt [rho e]   dx    [rho e]    dx [P v]

the 1st and 2nd terms are integrated via the flux integration
the 1st and 3rd terms are integrated via the pressure integration
	that is split into first the momentum and then the work diffusion 
*/

//the Hydrodynamics II book I went by said to use cell values for interface ...
//but now that I'm introducing boundary conditions, I'll have to cull cells based on whether they have boundaries
// and it's just easier to cull cell boundaries themselves.
// so I'm going to try this based on cell boundaries instead of cells.
__kernel void calcCFL(
	__global real* cflBuffer,
	const __global real* stateBuffer,
	const __global real* potentialBuffer,
	const __global char* boundaryBuffer,
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
	int indexL = index;

	real result = INFINITY;
	for (int side = 0; side < DIM; ++side) {
		int indexR = index + stepsize[side];

		const __global real* stateL = stateBuffer + NUM_STATES * indexL;
		const __global real* stateR = stateBuffer + NUM_STATES * indexR;

		//Roe averages?  regular averages?  no averages and just use cells instead of interfaces?
		
		real densityL = stateL[STATE_DENSITY];
		real4 velocityL = VELOCITY(stateL);
		real energyTotalL = stateL[STATE_ENERGY_TOTAL];
		
		real densityR = stateR[STATE_DENSITY];
		real4 velocityR = VELOCITY(stateR);
		real energyTotalR = stateR[STATE_ENERGY_TOTAL];

		real density = .5f * (densityL + densityR);
		real4 velocity = .5f * (velocityL + velocityR);
		real energyTotal = .5f * (energyTotalL + energyTotalR);
		
		real specificEnergyTotal = energyTotal / density;
		real specificEnergyKinetic = .5f * dot(velocity, velocity);
		real specificEnergyPotential = potentialBuffer[index];
		real specificEnergyInternal = specificEnergyTotal - specificEnergyKinetic - specificEnergyPotential;

		real speedOfSound = sqrt(gamma * (gamma - 1.f) * specificEnergyInternal);

#if DIM == 1
		real dim = dx[0] / (speedOfSound + fabs(velocity));
		result = min(result, dum);
#else
		for (int side = 0; side < DIM; ++side) {
			real dum = dx[side] / (speedOfSound + fabs(velocity[side]));
			result = min(result, dum);
		}
#endif
	}
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
	const __global real* state = stateBuffer + NUM_STATES * index;

	real density = state[STATE_DENSITY];
	real energyTotal = state[STATE_ENERGY_TOTAL];
	real4 velocity = VELOCITY(state);
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
		const __global real* stateL = stateBuffer + NUM_STATES * indexL;
		const __global real* stateR = stateBuffer + NUM_STATES * indexR;
		real velocityL = stateL[side+STATE_MOMENTUM_X];
		real velocityR = stateR[side+STATE_MOMENTUM_X];
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

