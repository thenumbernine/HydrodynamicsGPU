#include "HydroGPU/Shared/Common.h"

void calcEigenvaluesSide(
	__global real* eigenvaluesBuffer,
	const __global real* stateBuffer,
	const __global real* potentialBuffer,
	int side);

//I'm wondering if I could move a few calculations up here to before the dt calculation
//eigenvalues are needed for the cfl calculation
//and the cfl calculation is needed for the dt value
//dt is needed for Burgers for the 2nd order stuff in the flux calculations
// but not necessarily for the HLL/C solvers?  unless I finally added 2nd order slope limiters to HLL/C, maybe?
//
void calcEigenvaluesSide(
	__global real* eigenvaluesBuffer,
	const __global real* stateBuffer,
	const __global real* potentialBuffer,
	int side)
{
	int4 i = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);
	int index = INDEXV(i);
	int indexPrev = index - stepsize[side];
	int interfaceIndex = side + DIM * index;

	const __global real* srcStateL = stateBuffer + NUM_STATES * indexPrev;
	const __global real* srcStateR = stateBuffer + NUM_STATES * index;

	real stateL[NUM_STATES];
	real stateR[NUM_STATES];

	// rotate into x axis
	for (int i = 0; i < NUM_STATES; ++i) {
		stateL[i] = srcStateL[i];
		stateR[i] = srcStateR[i];
	}
	{
		real tmp;
		
		tmp = stateL[STATE_MOMENTUM_X];
		stateL[STATE_MOMENTUM_X] = stateL[STATE_MOMENTUM_X+side];
		stateL[STATE_MOMENTUM_X+side] = tmp;
		
		tmp = stateR[STATE_MOMENTUM_X];
		stateR[STATE_MOMENTUM_X] = stateR[STATE_MOMENTUM_X+side];
		stateR[STATE_MOMENTUM_X+side] = tmp;
	}

	__global real* eigenvalues = eigenvaluesBuffer + NUM_STATES * interfaceIndex;

	//the left and right primitives are recalculated in calcEigenvalues and in calcFlux
	//if they could be stored somewhere, that might speed things up, but would take up a bit more memory

	real densityL = stateL[STATE_DENSITY];
	real invDensityL = 1. / densityL;
	real4 velocityL = VELOCITY(stateL);
	real velocitySqL = dot(velocityL, velocityL);
	real energyTotalL = stateL[STATE_ENERGY_TOTAL] * invDensityL;
	real energyKineticL = .5 * velocitySqL;
	real energyPotentialL = potentialBuffer[indexPrev];
	real energyInternalL = energyTotalL - energyKineticL - energyPotentialL;
	real pressureL = (gamma - 1.) * densityL * energyInternalL;
	real enthalpyTotalL = energyTotalL + pressureL * invDensityL;
	real speedOfSoundL = sqrt((gamma - 1.) * (enthalpyTotalL - .5 * velocitySqL));
	real roeWeightL = sqrt(densityL);

	real densityR = stateR[STATE_DENSITY];
	real invDensityR = 1. / densityR;
	real4 velocityR = VELOCITY(stateR);
	real velocitySqR = dot(velocityR, velocityR);
	real energyTotalR = stateR[STATE_ENERGY_TOTAL] * invDensityR;
	real energyKineticR = .5 * velocitySqR;
	real energyPotentialR = potentialBuffer[index];
	real energyInternalR = energyTotalR - energyKineticR - energyPotentialR;
	real pressureR = (gamma - 1.) * densityR * energyInternalR;
	real enthalpyTotalR = energyTotalR + pressureR * invDensityR;
	real speedOfSoundR = sqrt((gamma - 1.) * (enthalpyTotalR - .5 * velocitySqR));
	real roeWeightR = sqrt(densityR);
	
	real roeWeightNormalization = 1. / (roeWeightL + roeWeightR);
	real4 velocity = (roeWeightL * velocityL + roeWeightR * velocityR) * roeWeightNormalization;
	real enthalpyTotal = (roeWeightL * enthalpyTotalL + roeWeightR * enthalpyTotalR) * roeWeightNormalization;
	real energyPotential = (roeWeightL * energyPotentialL + roeWeightR * energyPotentialR) * roeWeightNormalization; 
	
	real velocitySq = dot(velocity, velocity);
	real speedOfSound = sqrt((gamma - 1.) * (enthalpyTotal - .5 * velocitySq - energyPotential));

	//eigenvalues

	real eigenvaluesMinL = velocityL.x - speedOfSoundL;
	real eigenvaluesMaxR = velocityR.x + speedOfSoundR;
	real eigenvaluesMin = velocity.x - speedOfSound;
	real eigenvaluesMax = velocity.x + speedOfSound;
#if 0	//Davis direct
	real sl = eigenvaluesMinL;
	real sr = eigenvaluesMaxR;
#endif
#if 1	//Davis direct bounded
	real sl = min(eigenvaluesMinL, eigenvaluesMin);
	real sr = max(eigenvaluesMaxR, eigenvaluesMax);
#endif


	eigenvalues[0] = sl;
	eigenvalues[1] = velocity.x;
#if DIM > 1
	eigenvalues[2] = velocity.x;
#if DIM > 2
	eigenvalues[3] = velocity.x;
#endif
#endif
	eigenvalues[DIM+1] = sr;
}

__kernel void calcEigenvalues(
	__global real* eigenvaluesBuffer,
	const __global real* stateBuffer,
	const __global real* potentialBuffer)
{
	int4 i = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);
	if (i.x < 2 || i.x >= SIZE_X - 1
#if DIM > 1
		|| i.y < 2 || i.y >= SIZE_Y - 1
#endif
#if DIM > 2
		|| i.z < 2 || i.z >= SIZE_Z - 1
#endif
	) return;
	calcEigenvaluesSide(eigenvaluesBuffer, stateBuffer, potentialBuffer, 0);
#if DIM > 1
	calcEigenvaluesSide(eigenvaluesBuffer, stateBuffer, potentialBuffer, 1);
#endif
#if DIM > 2
	calcEigenvaluesSide(eigenvaluesBuffer, stateBuffer, potentialBuffer, 2);
#endif
}

//do we want to use interface wavespeeds to calculate cfl?
// esp when the flux values are computed from cell wavespeeds
__kernel void calcCellTimestep(
	__global real* dtBuffer,
	const __global real* eigenvaluesBuffer)
{
	int4 i = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);
	int index = INDEXV(i);
	if (i.x < 2 || i.x >= SIZE_X - 1 
#if DIM > 1
		|| i.y < 2 || i.y >= SIZE_Y - 1
#endif
#if DIM > 2
		|| i.z < 2 || i.z >= SIZE_Z - 1
#endif
	) {
		dtBuffer[index] = INFINITY;
		return;
	}
	
	real result = INFINITY;
	for (int side = 0; side < DIM; ++side) {
		int indexNext = index + stepsize[side];
		
		const __global real* eigenvaluesL = eigenvaluesBuffer + NUM_STATES * (side + DIM * index);
		const __global real* eigenvaluesR = eigenvaluesBuffer + NUM_STATES * (side + DIM * indexNext);
		
		real minLambda = min(0., eigenvaluesR[0]);
		real maxLambda = max(0., eigenvaluesL[DIM+1]);
		
		real dum = dx[side] / (maxLambda - minLambda);
		result = min(result, dum);
	}
	
	dtBuffer[index] = result;
}

void calcFluxSide(
	__global real* fluxBuffer,
	const __global real* stateBuffer,
	const __global real* eigenvaluesBuffer,
	const __global real* potentialBuffer,
	real dt_dx,
	int side);

void calcFluxSide(
	__global real* fluxBuffer,
	const __global real* stateBuffer,
	const __global real* eigenvaluesBuffer,
	const __global real* potentialBuffer,
	real dt_dx,
	int side)
{
	int4 i = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);
	int index = INDEXV(i);
	int indexPrev = index - stepsize[side];
	int interfaceIndex = side + DIM * index;

	const __global real* srcStateL = stateBuffer + NUM_STATES * indexPrev;
	const __global real* srcStateR = stateBuffer + NUM_STATES * index;

	real stateL[NUM_STATES];
	real stateR[NUM_STATES];

	// rotate into x axis
	for (int i = 0; i < NUM_STATES; ++i) {
		stateL[i] = srcStateL[i];
		stateR[i] = srcStateR[i];
	}
	{
		real tmp;
		
		tmp = stateL[STATE_MOMENTUM_X];
		stateL[STATE_MOMENTUM_X] = stateL[STATE_MOMENTUM_X+side];
		stateL[STATE_MOMENTUM_X+side] = tmp;
		
		tmp = stateR[STATE_MOMENTUM_X];
		stateR[STATE_MOMENTUM_X] = stateR[STATE_MOMENTUM_X+side];
		stateR[STATE_MOMENTUM_X+side] = tmp;
	}
	
	const __global real* eigenvalues = eigenvaluesBuffer + NUM_STATES * interfaceIndex;

	__global real* flux = fluxBuffer + NUM_FLUX_STATES * interfaceIndex;

	real densityL = stateL[STATE_DENSITY];
	real invDensityL = 1. / densityL;
	real4 velocityL = VELOCITY(stateL);
	real velocitySqL = dot(velocityL, velocityL);
	real energyTotalL = stateL[STATE_ENERGY_TOTAL] * invDensityL;
	real energyKineticL = .5 * velocitySqL;
	real energyPotentialL = potentialBuffer[indexPrev];
	real energyInternalL = energyTotalL - energyKineticL - energyPotentialL;
	real pressureL = (gamma - 1.) * densityL * energyInternalL;
	real enthalpyTotalL = energyTotalL + pressureL * invDensityL;

	real densityR = stateR[STATE_DENSITY];
	real invDensityR = 1. / densityR;
	real4 velocityR = VELOCITY(stateR);
	real velocitySqR = dot(velocityR, velocityR);
	real energyTotalR = stateR[STATE_ENERGY_TOTAL] * invDensityR;
	real energyKineticR = .5 * velocitySqR;
	real energyPotentialR = potentialBuffer[index];
	real energyInternalR = energyTotalR - energyKineticR - energyPotentialR;
	real pressureR = (gamma - 1.) * densityR * energyInternalR;
	real enthalpyTotalR = energyTotalR + pressureR * invDensityR;
	
	real sl = eigenvalues[0];
	real sr = eigenvalues[NUM_STATES-1];

	//flux

	real fluxL[NUM_STATES];
	fluxL[0] = densityL * velocityL.x;
	fluxL[1] = densityL * velocityL.x * velocityL.x +  pressureL;
#if DIM > 1
	fluxL[2] = densityL * velocityL.y * velocityL.x;
#endif
#if DIM > 2
	fluxL[3] = densityL * velocityL.z * velocityL.x;
#endif
	fluxL[DIM+1] = densityL * enthalpyTotalL * velocityL.x;
	
	real fluxR[NUM_STATES];
	fluxR[0] = densityR * velocityR.x;
	fluxR[1] = densityR * velocityR.x * velocityR.x + pressureR;
#if DIM > 1
	fluxR[2] = densityR * velocityR.y * velocityR.x;
#endif
#if DIM > 2
	fluxR[3] = densityR * velocityR.z * velocityR.x;
#endif
	fluxR[DIM+1] = densityR * enthalpyTotalR * velocityR.x;	

	//HLL-specific
	if (0. <= sl) {
		for (int i = 0; i < NUM_STATES; ++i) {
			flux[i] = fluxL[i];
		}
	} else if (sl <= 0. && 0. <= sr) {
		//(sr * fluxL[j] - sl * fluxR[j] + sl * sr * (stateR[j] - stateL[j])) / (sr - sl)
		real invDenom = 1. / (sr - sl);
		for (int i = 0; i < NUM_STATES; ++i) {
			flux[i] = (sr * fluxL[i] - sl * fluxR[i] + sl * sr * (stateR[i] - stateL[i])) * invDenom; 
		}
	} else if (sr <= 0.) {
		for (int i = 0; i < NUM_STATES; ++i) {
			flux[i] = fluxR[i];
		}
	}

/*
slope limiter?

theta = sign of eigenvalues

fhalf = 1/2 (fR + fL) - 1/2 (theta + phi (epsilon - theta)) (fR - fL)

donor cell when phi = 0:

fhalf = 1/2 (fR + fL) - 1/2 theta (fR - fL)
... for theta = 1: 	fhalf = 1/2 fR + 1/2 fL - 1/2 fR + 1/2 fL = fL
... for theta = -1:	fhalf = 1/2 fR + 1/2 fL + 1/2 fR - 1/2 fL = fR

... for phi = slope of rTile
			= for theta = 1:	delta q(i-1) / delta q(i)
			= for theta = -1:	delta q(i+1) / delta q(i)

so what do we use to consider delta q?  the states themselves, or any sort of transformation (as Roe does)?
how about (by Hydrodynamics ii) delta q = delta f / lambda
so what is delta f?  
Hydrodynamics ii stops at mentioning the deconstruction of delta q into delta q- and delta q+
 and its association with lambda+ and lambda-
 which are the eigenvalue min and max used in the flux calculation.
 is this the slope that should be used with the slope limiter?
or can we use delta q- and delta q+ for the lhs and rhs of the delta q slope, choosing one or the other based on the velocity sign 
*/
#if 0
	for (int i = 0; i < NUM_STATES; ++i) {
		real eigenvalue = eigenvalues[i];
		real deltaFlux = fluxR[i] - fluxL[i];
		real deltaQ = stateR[i] - stateL[i];
		real rTilde = deltaFlux / deltaQ;
		real theta;
		if (eigenvalue >= 0.) {
			theta = 1.;
			//rTilde = (stateMid - stateL[i]) / deltaQ;
		} else {
			theta = -1.;
			//rTilde = (stateR[i] - stateMid) / deltaQ;
		}
		real phi = slopeLimiter(rTilde);
		real epsilon = eigenvalue * dt_dx;
		flux[i] -= .5 * deltaFlux * (theta + phi * (epsilon - theta) / (real)DIM);
	}
#endif

	//rotate back to side
	{
		real tmp = flux[STATE_MOMENTUM_X];
		flux[STATE_MOMENTUM_X] = flux[STATE_MOMENTUM_X + side];
		flux[STATE_MOMENTUM_X + side] = tmp;
	}
}

__kernel void calcFlux(
	__global real* fluxBuffer,
	const __global real* stateBuffer,
	const __global real* eigenvaluesBuffer,
	const __global real* potentialBuffer,
	real dt)
{
	int4 i = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);
	if (i.x < 2 || i.x >= SIZE_X - 1
#if DIM > 1
		|| i.y < 2 || i.y >= SIZE_Y - 1
#endif
#if DIM > 2
		|| i.z < 2 || i.z >= SIZE_Z - 1
#endif
	) return;
	
	calcFluxSide(fluxBuffer, stateBuffer, eigenvaluesBuffer, potentialBuffer, dt/DX, 0);
#if DIM > 1
	calcFluxSide(fluxBuffer, stateBuffer, eigenvaluesBuffer, potentialBuffer, dt/DY, 1);
#endif
#if DIM > 2
	calcFluxSide(fluxBuffer, stateBuffer, eigenvaluesBuffer, potentialBuffer, dt/DZ, 2);
#endif
}
