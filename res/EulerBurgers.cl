#include "HydroGPU/Shared/Common.h"
#include "HydroGPU/Euler.h"

#define gamma idealGas_heatCapacityRatio	//laziness

/*
Euler Burgers:
 
  d [ rho ]    d    [ rho ]     d [ 0 ]
  - [rho v] +  - (v [rho v]) +  - [ P ] = 0
 dt [rho e]   dx    [rho e]    dx [P v]

the 1st and 2nd terms are integrated via the flux integration
the 1st and 3rd terms are integrated via the pressure integration
	that is split into first the momentum and then the work diffusion 
*/

//based on max inter-cell wavespeed
__kernel void calcCellTimestep(
	__global real* dtBuffer,
	const __global real* stateBuffer,
	const __global real* potentialBuffer
#ifdef SOLID
	, const __global char* solidBuffer
#endif
	)
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
		dtBuffer[index] = INFINITY;
		return;
	}

#ifdef SOLID
	if (solidBuffer[index]) {
		dtBuffer[index] = INFINITY;
		return;
	}
#endif

	const __global real* state = stateBuffer + NUM_STATES * index;
	real density = state[STATE_DENSITY];
	real energyTotal = state[STATE_ENERGY_TOTAL];
	
	real velocityX = state[STATE_MOMENTUM_X] / density;
	real velocitySq = velocityX * velocityX;
#if DIM > 1
	real velocityY = state[STATE_MOMENTUM_Y] / density;
	velocitySq += velocityY * velocityY;
#if DIM > 2
	real velocityZ = state[STATE_MOMENTUM_Z] / density;
	velocitySq += velocityZ * velocityZ;
#endif
#endif

	real specificEnergyTotal = energyTotal / density;
	real specificEnergyKinetic = .5 * velocitySq;
	real specificEnergyPotential = potentialBuffer[index];
	real specificEnergyInternal = specificEnergyTotal - specificEnergyKinetic - specificEnergyPotential;

	real speedOfSound = sqrt(gamma * (gamma - 1.) * specificEnergyInternal);
	real result = DX / (speedOfSound + fabs(velocityX));
#if DIM > 1
	result = min(result, DY / (speedOfSound + fabs(velocityY)));
#if DIM > 2
	result = min(result, DZ / (speedOfSound + fabs(velocityZ)));
#endif
#endif
	dtBuffer[index] = result;
}

void calcInterfaceVelocitySide(
	__global real* interfaceVelocityBuffer,
	const __global real* stateBuffer,
#ifdef SOLID
	const __global char* solidBuffer,
#endif
	int side);

void calcInterfaceVelocitySide(
	__global real* interfaceVelocityBuffer,
	const __global real* stateBuffer,
#ifdef SOLID
	const __global char* solidBuffer,
#endif
	int side)
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
	int indexL = index - stepsize[side];

	int interfaceIndex = side + DIM * index;

	real densityL = stateBuffer[STATE_DENSITY + NUM_STATES * indexL];
	real velocityL = stateBuffer[side+STATE_MOMENTUM_X + NUM_STATES * indexL] / densityL;
	
	real densityR = stateBuffer[STATE_DENSITY + NUM_STATES * indexR];
	real velocityR = stateBuffer[side+STATE_MOMENTUM_X + NUM_STATES * indexR] / densityR;

#ifdef SOLID
	char solidL = solidBuffer[indexL];
	char solidR = solidBuffer[indexR];
	if (solidL && !solidR) {
		velocityL = -velocityR;
	} else if (solidR && !solidL) {
		velocityR = -velocityL;
	}
#endif

	interfaceVelocityBuffer[interfaceIndex] = .5 * (velocityL + velocityR);
}

__kernel void calcInterfaceVelocity(
	__global real* interfaceVelocityBuffer,
	const __global real* stateBuffer
#ifdef SOLID
	, const __global char* solidBuffer
#endif
)
{
	for (int side = 0; side < DIM; ++side) {
		calcInterfaceVelocitySide(interfaceVelocityBuffer, stateBuffer,
#ifdef SOLID
			solidBuffer,
#endif
			side);
	}
}

void calcFluxSide(
	__global real* fluxBuffer,
	const __global real* stateBuffer,
	const __global real* interfaceVelocityBuffer,
#ifdef SOLID
	const __global char* solidBuffer,	
#endif
	real dt,
	int side);

void calcFluxSide(
	__global real* fluxBuffer,
	const __global real* stateBuffer,
	const __global real* interfaceVelocityBuffer,
#ifdef SOLID
	const __global char* solidBuffer,	
#endif
	real dt,
	int side)
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

	real4 dt_dx = dt / dx;

	int index = INDEXV(i);
	int indexR = index;

	int indexL = index - stepsize[side];
	int indexL2 = indexL - stepsize[side];
	int indexR2 = index + stepsize[side];

#ifdef SOLID
	char solidL2 = solidBuffer[indexL2];
	char solidL = solidBuffer[indexL];
	char solidR = solidBuffer[indexR];
	char solidR2 = solidBuffer[indexR2];
#endif

	int interfaceIndex = side + DIM * index;

	real interfaceVelocity = interfaceVelocityBuffer[interfaceIndex];

	__global real* flux = fluxBuffer + NUM_FLUX_STATES * interfaceIndex;

	for (int j = 0; j < NUM_FLUX_STATES; ++j) {
		real stateR2 = stateBuffer[j + NUM_STATES * indexR2];
		real stateR = stateBuffer[j + NUM_STATES * indexR];
		real stateL = stateBuffer[j + NUM_STATES * indexL];
		real stateL2 = stateBuffer[j + NUM_STATES * indexL2];

#ifdef SOLID
		/*
		slope limiters and arbitrary boundaries...
		we have to look two over in any direction
		if there's a wall two over, what do we do?
		*/
		if (solidL && solidR) {	//shouldn't be anything anyways ...
		} else if (solidL) {
			if (j == STATE_MOMENTUM_X+side) {
				stateL = stateR;
			} else {
				stateL = -stateR;
			}
		} else if (solidR) {
			if (j == STATE_MOMENTUM_X+side) {
				stateR = stateL;
			} else {
				stateR = -stateL;
			}
		}
		
		if (solidL2) {
			stateL2 = stateL;
			if (j == STATE_MOMENTUM_X+side) stateL2 = -stateL2;
		}
		if (solidR2) {
			stateR2 = stateR;
			if (j == STATE_MOMENTUM_X+side) stateR2 = -stateR2;
		}
#endif

		real deltaStateL = stateL - stateL2;
		real deltaState = stateR - stateL;
		real deltaStateR = stateR2 - stateR;

		//3D case crashes?
		//real flux = mix(stateR, stateL, theta) * interfaceVelocity;

		//this line crashes when compiling on my Intel HD4000 only for the 3D case
		//real stateSlopeRatio = mix(deltaStateR, deltaStateL, theta) / deltaState;
		//...but writing it out explicitly works fine
		real theta;
		real stateSlopeRatio;
		if (interfaceVelocity >= 0.) {
			theta = 1.;
			stateSlopeRatio = deltaStateL / deltaState;
		} else {
			theta = -1.;
			stateSlopeRatio = deltaStateR / deltaState;
		}
		//2nd order stuff:
		real phi = slopeLimiter(stateSlopeRatio);
		
		flux[j] = .5 * interfaceVelocity * ((1. + theta) * stateL + (1. - theta) * stateR)
				+ .5 * deltaState * phi * fabs(interfaceVelocity) * (1. - fabs(interfaceVelocity * dt_dx[side]));
		//		/ (real)DIM;	//this wasn't in the Hydrodynamics II papers, but it seems to help.  there is some error with splitting higher dimensions.
	}
}

__kernel void calcFlux(
	__global real* fluxBuffer,
	const __global real* stateBuffer,
	const __global real* interfaceVelocityBuffer,
#ifdef SOLID
	const __global char* solidBuffer,	
#endif
	real dt)
{
	for (int side = 0; side < DIM; ++side) {
		calcFluxSide(fluxBuffer, stateBuffer, interfaceVelocityBuffer
#ifdef SOLID
			, solidBuffer
#endif
			, dt, side);
	}
}

__kernel void computePressure(
	__global real* pressureBuffer,
	const __global real* stateBuffer,
	const __global real* potentialBuffer
#ifdef SOLID
	, const __global char* solidBuffer
#endif
	)
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

#ifdef SOLID
	if (solidBuffer[index]) return;
#endif

	const __global real* state = stateBuffer + NUM_STATES * index;

	real density = state[STATE_DENSITY];
	real energyTotal = state[STATE_ENERGY_TOTAL];
	
	real velocityX = state[STATE_MOMENTUM_X] / density;
	real velocitySq = velocityX * velocityX;
#if DIM > 1
	real velocityY = state[STATE_MOMENTUM_Y] / density;
	velocitySq += velocityY * velocityY;
#if DIM > 2
	real velocityZ = state[STATE_MOMENTUM_Z] / density;
	velocitySq += velocityZ * velocityZ;
#endif
#endif

	real specificEnergyTotal = energyTotal / density;
	real specificEnergyKinetic = .5 * velocitySq;
	real specificEnergyPotential = potentialBuffer[index];
	real specificEnergyInternal = specificEnergyTotal - specificEnergyKinetic - specificEnergyPotential;
	real pressure = (gamma - 1.) * density * specificEnergyInternal;

#ifdef USE_VON_NEUMANN_RICHTMYER_ARTIFICIAL_VISCOSITY
	//TODO I broke something
	//von Neumann-Richtmyer artificial viscosity
	real deltaVelocityX = stateBuffer[STATE_MOMENTUM_X + NUM_STATES * (index+STEP_X)] / stateBuffer[STATE_DENSITY + NUM_STATES * (index+STEP_X)]
						- stateBuffer[STATE_MOMENTUM_X + NUM_STATES * (index-STEP_X)] / stateBuffer[STATE_DENSITY + NUM_STATES * (index-STEP_X)];
	real deltaVelocitySq = deltaVelocityX * deltaVelocityX; 
#if DIM > 1
	real deltaVelocityY = stateBuffer[STATE_MOMENTUM_Y + NUM_STATES * (index+STEP_Y)] / stateBuffer[STATE_DENSITY + NUM_STATES * (index+STEP_Y)]
						- stateBuffer[STATE_MOMENTUM_Y + NUM_STATES * (index-STEP_Y)] / stateBuffer[STATE_DENSITY + NUM_STATES * (index-STEP_Y)];
	deltaVelocitySq += deltaVelocityY * deltaVelocityY; 
#if DIM > 2
	real deltaVelocityZ = stateBuffer[STATE_MOMENTUM_Z + NUM_STATES * (index+STEP_Z)] / stateBuffer[STATE_DENSITY + NUM_STATES * (index+STEP_Z)]
						- stateBuffer[STATE_MOMENTUM_Z + NUM_STATES * (index-STEP_Z)] / stateBuffer[STATE_DENSITY + NUM_STATES * (index-STEP_Z)];
	deltaVelocitySq += deltaVelocityZ * deltaVelocityZ; 
#endif
#endif
	const real ZETA = 2.;
	pressure += .25 * ZETA * ZETA * density * deltaVelocitySq;
#endif	//USE_VON_NEUMANN_RICHTMYER_ARTIFICIAL_VISCOSITY

	pressureBuffer[index] = pressure;
}

__kernel void diffuseMomentum(
	__global real* derivBuffer,
	const __global real* pressureBuffer
#ifdef SOLID
	, const __global char* solidBuffer
#endif
	)
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

#ifdef SOLID
	if (solidBuffer[index]) return;
#endif
	
	__global real* deriv = derivBuffer + NUM_STATES * index;

	real pressureL, pressureR;

	pressureL = pressureBuffer[index - STEP_X];
	pressureR = pressureBuffer[index + STEP_X];
#ifdef SOLID
	if (solidBuffer[index - STEP_X]) pressureL = pressureBuffer[index];
	if (solidBuffer[index + STEP_X]) pressureR = pressureBuffer[index];
#endif
	deriv[STATE_MOMENTUM_X] -= .5 * (pressureR - pressureL) / DX;

#if DIM > 1
	pressureL = pressureBuffer[index - STEP_Y];
	pressureR = pressureBuffer[index + STEP_Y];
#ifdef SOLID
	if (solidBuffer[index - STEP_Y]) pressureL = pressureBuffer[index];
	if (solidBuffer[index + STEP_Y]) pressureR = pressureBuffer[index];
#endif
	deriv[STATE_MOMENTUM_Y] -= .5 * (pressureR - pressureL) / DY;

#if DIM > 2
	pressureL = pressureBuffer[index - STEP_Z];
	pressureR = pressureBuffer[index + STEP_Z];
#ifdef SOLID
	if (solidBuffer[index - STEP_Z]) pressureL = pressureBuffer[index];
	if (solidBuffer[index + STEP_Z]) pressureR = pressureBuffer[index];
#endif
	deriv[STATE_MOMENTUM_Z] -= .5 * (pressureR - pressureL) / DZ;
#endif
#endif
}

__kernel void diffuseWork(
	__global real* derivBuffer,
	const __global real* stateBuffer,
	const __global real* pressureBuffer
#ifdef SOLID
	, const __global char* solidBuffer
#endif
	)
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

#ifdef SOLID
	if (solidBuffer[index]) return;
#endif
	__global real* deriv = derivBuffer + NUM_STATES * index;

	real velocityL, velocityR, pressureL, pressureR;
	real deltaEnergyTotal = 0.;

	velocityL = stateBuffer[STATE_MOMENTUM_X + NUM_STATES * (index-STEP_X)] / stateBuffer[STATE_DENSITY + NUM_STATES * (index-STEP_X)];
	velocityR = stateBuffer[STATE_MOMENTUM_X + NUM_STATES * (index+STEP_X)] / stateBuffer[STATE_DENSITY + NUM_STATES * (index+STEP_X)];
	pressureL = pressureBuffer[index-STEP_X];
	pressureR = pressureBuffer[index+STEP_X];
#ifdef SOLID
	if (solidBuffer[index-STEP_X]) velocityL = -velocityL;
	if (solidBuffer[index+STEP_X]) velocityR = -velocityR;
	if (solidBuffer[index-STEP_X]) pressureL = pressureBuffer[index];
	if (solidBuffer[index+STEP_X]) pressureR = pressureBuffer[index];
#endif
	deltaEnergyTotal -= .5 * (pressureR * velocityR - pressureL * velocityL) / DX;

#if DIM > 1
	velocityL = stateBuffer[STATE_MOMENTUM_Y + NUM_STATES * (index-STEP_Y)] / stateBuffer[STATE_DENSITY + NUM_STATES * (index-STEP_Y)];
	velocityR = stateBuffer[STATE_MOMENTUM_Y + NUM_STATES * (index+STEP_Y)] / stateBuffer[STATE_DENSITY + NUM_STATES * (index+STEP_Y)];
	pressureL = pressureBuffer[index-STEP_Y];
	pressureR = pressureBuffer[index+STEP_Y];
#ifdef SOLID
	if (solidBuffer[index-STEP_Y]) velocityL = -velocityL;
	if (solidBuffer[index+STEP_Y]) velocityR = -velocityR;
	if (solidBuffer[index-STEP_Y]) pressureL = pressureBuffer[index];
	if (solidBuffer[index+STEP_Y]) pressureR = pressureBuffer[index];
#endif
	deltaEnergyTotal -= .5 * (pressureR * velocityR - pressureL * velocityL) / DY;

#if DIM > 2
	velocityL = stateBuffer[STATE_MOMENTUM_Z + NUM_STATES * (index-STEP_Z)] / stateBuffer[STATE_DENSITY + NUM_STATES * (index-STEP_Z)];
	velocityR = stateBuffer[STATE_MOMENTUM_Z + NUM_STATES * (index+STEP_Z)] / stateBuffer[STATE_DENSITY + NUM_STATES * (index+STEP_Z)];
	pressureL = pressureBuffer[index-STEP_Z];
	pressureR = pressureBuffer[index+STEP_Z];
#ifdef SOLID
	if (solidBuffer[index-STEP_Z]) velocityL = -velocityL;
	if (solidBuffer[index+STEP_Z]) velocityR = -velocityR;
	if (solidBuffer[index-STEP_Z]) pressureL = pressureBuffer[index];
	if (solidBuffer[index+STEP_Z]) pressureR = pressureBuffer[index];
#endif
	deltaEnergyTotal -= .5 * (pressureR * velocityR - pressureL * velocityL) / DZ;

#endif
#endif

	deriv[STATE_ENERGY_TOTAL] += deltaEnergyTotal; 
}
