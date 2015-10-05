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

//based on max inter-cell wavespeed
__kernel void findMinTimestep(
	__global real* dtBuffer,
	const __global real* stateBuffer,
	const __global real* potentialBuffer,
	const __global char* solidBuffer)
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

	if (solidBuffer[index]) {
		dtBuffer[index] = INFINITY;
		return;
	}

	const __global real* state = stateBuffer + NUM_STATES * index;
	real density = state[STATE_DENSITY];
	real energyTotal = state[STATE_ENERGY_TOTAL];
	real4 velocity = VELOCITY(state);
	
	real specificEnergyTotal = energyTotal / density;
	real specificEnergyKinetic = .5f * dot(velocity, velocity);
	real specificEnergyPotential = potentialBuffer[index];
	real specificEnergyInternal = specificEnergyTotal - specificEnergyKinetic - specificEnergyPotential;

	real speedOfSound = sqrt(gamma * (gamma - 1.f) * specificEnergyInternal);
	real result = dx[0] / (speedOfSound + fabs(velocity[0]));
	for (int side = 1; side < DIM; ++side) {
		real dum = dx[side] / (speedOfSound + fabs(velocity[side]));
		result = min(result, dum);
	}
	dtBuffer[index] = result;
}

__kernel void calcInterfaceVelocity(
	__global real* interfaceVelocityBuffer,
	const __global real* stateBuffer,
	const __global char* solidBuffer,
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
		
	real densityL = stateBuffer[STATE_DENSITY + NUM_STATES * indexL];
	real velocityL = stateBuffer[side+STATE_MOMENTUM_X + NUM_STATES * indexL] / densityL;
	
	real densityR = stateBuffer[STATE_DENSITY + NUM_STATES * indexR];
	real velocityR = stateBuffer[side+STATE_MOMENTUM_X + NUM_STATES * indexR] / densityR;

	char solidL = solidBuffer[indexL];
	char solidR = solidBuffer[indexR];
	if (solidL && !solidR) {
		velocityL = -velocityR;
	} else if (solidR && !solidL) {
		velocityR = -velocityL;
	}
	
	interfaceVelocityBuffer[index] = .5f * (velocityL + velocityR);
}

__kernel void calcFlux(
	__global real* fluxBuffer,
	const __global real* stateBuffer,
	const __global real* interfaceVelocityBuffer,
	const __global char* solidBuffer,	
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

	char solidL2 = solidBuffer[indexL2];
	char solidL = solidBuffer[indexL];
	char solidR = solidBuffer[indexR];
	char solidR2 = solidBuffer[indexR2];

	real interfaceVelocity = interfaceVelocityBuffer[index];

	__global real* flux = fluxBuffer + NUM_STATES * index;

	for (int j = 0; j < NUM_STATES; ++j) {
		real stateR2 = stateBuffer[j + NUM_STATES * indexR2];
		real stateR = stateBuffer[j + NUM_STATES * indexR];
		real stateL = stateBuffer[j + NUM_STATES * indexL];
		real stateL2 = stateBuffer[j + NUM_STATES * indexL2];
		
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
		if (interfaceVelocity >= 0.f) {
			theta = 1.f;
			stateSlopeRatio = deltaStateL / deltaState;
		} else {
			theta = -1.f;
			stateSlopeRatio = deltaStateR / deltaState;
		}
		//2nd order stuff:
		real phi = slopeLimiter(stateSlopeRatio);
		
		flux[j] = .5f * interfaceVelocity * ((1.f + theta) * stateL + (1.f - theta) * stateR)
				+ .5f * deltaState * phi * fabs(interfaceVelocity) * (1.f - fabs(interfaceVelocity * dt_dx[side]));
		//		/ (real)DIM;	//this wasn't in the Hydrodynamics II papers, but it seems to help.  there is some error with splitting higher dimensions.
	}
}

__kernel void calcFluxDeriv(
	__global real* derivBuffer,	//dstate/dt
	const __global real* fluxBuffer,
	int side
#ifdef SOLID
	, const __global char* solidBuffer
#endif	//SOLID
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
#endif	//SOLID

	__global real* deriv = derivBuffer + NUM_STATES * index;

	int indexNext = index + stepsize[side];
	const __global real* fluxL = fluxBuffer + NUM_STATES * index;
	const __global real* fluxR = fluxBuffer + NUM_STATES * indexNext;
	for (int j = 0; j < NUM_STATES; ++j) {
		real deltaFlux = fluxR[j] - fluxL[j];
		deriv[j] -= deltaFlux / dx[side];
	}
}

__kernel void computePressure(
	__global real* pressureBuffer,
	const __global real* stateBuffer,
	const __global real* potentialBuffer,
	const __global char* solidBuffer)
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
	
	if (solidBuffer[index]) return;

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
	
	/*
	pressure = (gamma - 1) * density * specificEnergyInternal ... plus artificial viscosity ...
	pressure = (gamma - 1) * density * (specificEnergyTotal - specificEnergyKinetic - specificEnergyPotential)
	pressure = (gamma - 1) * (density * specificEnergyTotal - density * specificEnergyKinetic - density * specificEnergyPotential)
	pressure = (gamma - 1) * (energyTotal - .5 * (momentum * momentum) - density * specificEnergyPotential)
	pressure = [-(gamma - 1) * specificEnergyPotential] * density
				+ [-(gamma - 1) * .5 * momentum] * momentum
				+ [(gamma - 1)] * energyTotal
	...and the artificial viscosity component:
	
	viscosity = deltaVelocity * deltaVelocity
	viscosity = ZETA * .5 * (velocityR - velocityL)^2
	viscosity = [ZETA * .5 * deltaVelocity * density / densityR] * momentumR
				+ [-ZETA * .5 * deltaVelocity * density / densityL] * momentumL
	
	...or linearize it by density, nice and easy ...
	viscosity = [deltaVelocitySq] * density
	*/
	pressureBuffer[index] = pressure;
}

__kernel void diffuseMomentum(
	__global real* derivBuffer,
	const __global real* pressureBuffer,
	const __global char* solidBuffer)
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

	if (solidBuffer[index]) return;

	__global real* deriv = derivBuffer + NUM_STATES * index;

	for (int side = 0; side < DIM; ++side) {
		int indexL = index - stepsize[side];
		int indexR = index + stepsize[side];	

		real pressureC = pressureBuffer[index];
		
		real pressureL = pressureBuffer[indexL];
		if (solidBuffer[indexL]) pressureL = pressureC;

		real pressureR = pressureBuffer[indexR];
		if (solidBuffer[indexR]) pressureR = pressureC;

		real deltaPressure = .5f * (pressureR - pressureL);
		deriv[side + STATE_MOMENTUM_X] -= deltaPressure / dx[side];

		/*
		deriv[STATE_MOMENTUM_X+side] = -deltaPressure / dx[side]
									= (pressureL - pressureR) / dx[side]
		...pressureL and pressureR themselves use neighbor information when it comes to artificial viscosity, when linearizing it based on velocity...	
		so if we want to do that, we'll need a larger stencil for derivative state coefficients (2-wide rather than 1-wide) 
		deriv[STATE_MOMENTUM_X+side] = 
				//left cell:
				  [-(gamma - 1) * specificEnergyPotentialL / dx[side]] * densityL
				+ [-(gamma - 1) * .5 * momentum / dx[side]] * momentumL
				+ [(gamma - 1) / dx[side]] * energyTotalL
				//right cell:
				+ [(gamma - 1) * specificEnergyPotentialR / dx[side]] * densityR
				+ [(gamma - 1) * .5 * momentum / dx[side]] * momentumR
				+ [-(gamma - 1) / dx[side]] * energyTotalR
				//center cell:
				+ [ZETA * .5 * sum_j (velocityR[j] - velocityL[j])^2] * density
		*/
	}
}

__kernel void diffuseWork(
	__global real* derivBuffer,
	const __global real* stateBuffer,
	const __global real* pressureBuffer,
	const __global char* solidBuffer)
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

	if (solidBuffer[index]) return;

	__global real* deriv = derivBuffer + NUM_STATES * index;

	for (int side = 0; side < DIM; ++side) {
		int indexL = index - stepsize[side];
		int indexR = index + stepsize[side];

		real velocityL = stateBuffer[side+STATE_MOMENTUM_X + NUM_STATES * indexL] / stateBuffer[STATE_DENSITY + NUM_STATES * indexL];
		if (solidBuffer[indexL]) velocityL = -velocityL;

		real velocityR = stateBuffer[side+STATE_MOMENTUM_X + NUM_STATES * indexR] / stateBuffer[STATE_DENSITY + NUM_STATES * indexR];
		if (solidBuffer[indexR]) velocityR = -velocityR;
	
		real pressureC = pressureBuffer[index];

		real pressureL = pressureBuffer[indexL];
		if (solidBuffer[indexL]) pressureL = pressureC;

		real pressureR = pressureBuffer[indexR];
		if (solidBuffer[indexR]) pressureR = pressureC;

		real deltaWork = .5f * (pressureR * velocityR - pressureL * velocityL);

		deriv[STATE_ENERGY_TOTAL] -= deltaWork / dx[side];
	}
}

//this multiplies the derivative matrix -- sparse, but stored as a dense structure -- by the state buffer, unraveled as a vector
//dState/dt = A*State
//TODO should this go in Common.cl once I convert all the solvers over to support deriv coeffs / be implicit-compatible
__kernel void calcDerivFromStateCoeffs(
	__global real* derivBuffer,	//resulting dState/dt
	const __global real* stateBuffer,	//State
	const __global real* derivStateCoeffBuffer,	//dense structure of sparse matrix
	const __global char* solidBuffer,
	int side)
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

	if (solidBuffer[index]) return;
	
	const __global real* state = stateBuffer + NUM_STATES * index;
	__global real* deriv = derivBuffer + NUM_STATES * index;

#define NUM_NEIGHBORS (1 + 2 * DIM)
	const __global real* derivStateCoeffs = derivStateCoeffBuffer + NUM_STATES * NUM_NEIGHBORS * index;
	for (int j = 0; j < NUM_STATES; ++j) {
		deriv[j] += state[j] * derivStateCoeffs[j + NUM_STATES * 2 * DIM];
	}

	//for (int side = 0; side < DIM; ++side)
	{
		int indexPrev = index - stepsize[side];
		int indexNext = index + stepsize[side];
		const __global real* stateL = stateBuffer + NUM_STATES * indexPrev;
		const __global real* stateR = stateBuffer + NUM_STATES * indexNext;
		for (int j = 0; j < NUM_STATES; ++j) {
			deriv[j] += stateL[j] * derivStateCoeffs[j + NUM_STATES * (0 + 2 * side)];
			deriv[j] += stateR[j] * derivStateCoeffs[j + NUM_STATES * (1 + 2 * side)];
		}
	}
}

