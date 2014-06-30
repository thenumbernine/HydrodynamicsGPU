#include "HydroGPU/Shared/Common.h"

//flux
#if DIM == 1
#define FLUX(density, velocity, velocityN, pressure, enthalpyTotal, normal)	\
{	\
	density * velocityN,	\
	density * velocity.x * velocityN + pressure,	\
	density * enthalpyTotal * velocityN	\
}
#elif DIM == 2
#define FLUX(density, velocity, velocityN, pressure, enthalpyTotal, normal)	\
{	\
	density * velocityN,	\
	density * velocity.x * velocityN + pressure * normal.x,	\
	density * velocity.y * velocityN + pressure * normal.y,	\
	density * enthalpyTotal * velocityN	\
}
#elif DIM == 3
#define FLUX(density, velocity, velocityN, pressure, enthalpyTotal, normal)	\
{	\
	density * velocityN,	\
	density * velocity.x * velocityN + pressure * normal.x,	\
	density * velocity.y * velocityN + pressure * normal.y,	\
	density * velocity.z * velocityN + pressure * normal.z,	\
	density * enthalpyTotal * velocityN	\
}
#endif

__kernel void calcFluxAndEigenvalues(
	__global real* eigenvaluesBuffer,
	__global real* fluxBuffer,
	const __global real* stateBuffer,
	const __global real* gravityPotentialBuffer)
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
	int index = INDEXV(i);
	int indexR = index;
	for (int side = 0; side < DIM; ++side) {	
		int indexL = index - stepsize[side];
		int interfaceIndex = side + DIM * index;

		real4 normal = (real4)(0.f, 0.f, 0.f, 0.f);
		normal[side] = 1;

		const __global real* stateL = stateBuffer + NUM_STATES * indexL;
		const __global real* stateR = stateBuffer + NUM_STATES * indexR;

		real densityL = stateL[STATE_DENSITY];
		real invDensityL = 1.f / densityL;
		real4 velocityL = VELOCITY(stateL);
		real velocitySqL = dot(velocityL, velocityL);
		real velocityNL = dot(velocityL, normal);
		real energyTotalL = stateL[STATE_ENERGY_TOTAL] * invDensityL;
		real energyKineticL = .5f * velocitySqL;
		real energyPotentialL = gravityPotentialBuffer[indexL];
		real energyInternalL = energyTotalL - energyKineticL - energyPotentialL;
		real pressureL = (GAMMA - 1.f) * densityL * energyInternalL;
		real enthalpyTotalL = energyTotalL + pressureL * invDensityL;
		real weightL = sqrt(densityL);
		real speedOfSoundL = sqrt((GAMMA - 1.f) * (enthalpyTotalL - .5f * velocitySqL));
		real eigenvaluesMinL = velocityNL - speedOfSoundL;

		real densityR = stateR[STATE_DENSITY];
		real invDensityR = 1.f / densityR;
		real4 velocityR = VELOCITY(stateR);
		real velocitySqR = dot(velocityR, velocityR);
		real velocityNR = dot(velocityR, normal);
		real energyTotalR = stateR[STATE_ENERGY_TOTAL] * invDensityR;
		real energyKineticR = .5f * velocitySqR;
		real energyPotentialR = gravityPotentialBuffer[indexR];
		real energyInternalR = energyTotalR - energyKineticR - energyPotentialR;
		real pressureR = (GAMMA - 1.f) * densityR * energyInternalR;
		real enthalpyTotalR = energyTotalR + pressureR * invDensityR;
		real weightR = sqrt(densityR);
		real speedOfSoundR = sqrt((GAMMA - 1.f) * (enthalpyTotalR - .5f * velocitySqR));
		real eigenvaluesMaxR = velocityNR + speedOfSoundR;
		
		real roeWeightNormalization = 1.f / (weightL + weightR);
		real4 velocity = (weightL * velocityL + weightR * velocityR) * roeWeightNormalization;
		real enthalpyTotal = (weightL * enthalpyTotalL + weightR * enthalpyTotalR) * roeWeightNormalization;
		
		real velocitySq = dot(velocity, velocity);
		real speedOfSound = sqrt((GAMMA - 1.f) * (enthalpyTotal - .5f * velocitySq));
		real velocityN = dot(velocity, normal);
	
		//eigenvalues
	
		__global real* eigenvalues = eigenvaluesBuffer + NUM_STATES * interfaceIndex;

		eigenvalues[0] = velocityN - speedOfSound;
		eigenvalues[1] = velocityN;
#if DIM > 1
		eigenvalues[2] = velocityN;
#endif
#if DIM > 2
		eigenvalues[3] = velocityN;
#endif
		eigenvalues[DIM+1] = velocityN + speedOfSound;

		//flux

		real fluxL[NUM_STATES] = FLUX(densityL, velocityL, velocityNL, pressureL, enthalpyTotalL, normal);
		//the following line crashes the compiler
		real fluxR[NUM_STATES] = FLUX(densityR, velocityR, velocityNR, pressureR, enthalpyTotalR, normal);

#if 0	//Davis direct
		real sl = eigenvaluesMinL;
		real sr = eigenvaluesR[NUM_STATES-1];
#endif
#if 1	//Davis direct bounded
		real sl = min(eigenvaluesMinL, eigenvalues[0]);
		real sr = max(eigenvaluesMaxR, eigenvalues[NUM_STATES-1]);
#endif
	
		for (int j = 0; j < NUM_STATES; ++j) {
			if (sl >= 0.f) {
				fluxBuffer[j + NUM_STATES * interfaceIndex] = fluxL[j];
			} else if (sl <= 0.f && sr >= 0.f) {
				fluxBuffer[j + NUM_STATES * interfaceIndex] = (sr * fluxL[j] - sl * fluxR[j] + sl * sr * (stateR[j] - stateL[j])) / (sr - sl);
			} else if (sr <= 0.f) {
				fluxBuffer[j + NUM_STATES * interfaceIndex] = fluxR[j];
			}
		}
	}
}

//do we want to use interface wavespeeds to calculate cfl?
// esp when the flux values are computed from cell wavespeeds
__kernel void calcCFL(
	__global real* cflBuffer,
	const __global real* eigenvaluesBuffer,
	real cfl)
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
		cflBuffer[index] = INFINITY;
		return;
	}
	int indexL = index;

	real result = INFINITY;
	for (int side = 0; side < DIM; ++side) {
		int indexR = index + stepsize[side];
		
		const __global real* eigenvaluesL = eigenvaluesBuffer + NUM_STATES * (side + DIM * indexL);
		const __global real* eigenvaluesR = eigenvaluesBuffer + NUM_STATES * (side + DIM * indexR);

		real minLambda = 0.f;
		real maxLambda = 0.f;
		for (int i = 0; i < NUM_STATES; ++i) {	
			maxLambda = max(maxLambda, eigenvaluesL[i]);
			minLambda = min(minLambda, eigenvaluesR[i]);
		}

		real dum = dx[side] / (maxLambda - minLambda);
		result = min(result, dum);
	}
		
	cflBuffer[index] = cfl * result;
}

__kernel void integrateFlux(
	__global real* stateBuffer,
	const __global real* fluxBuffer,
	const __global real* dtBuffer)
{
	float dt = dtBuffer[0];
	real4 dt_dx = (real4)(dt / DX, dt / DY, dt / DZ, dt);

	int4 i = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);
	if (i.x < 2 || i.x >= SIZE_X - 1 
#if DIM > 1
		|| i.y < 2 || i.y >= SIZE_Y - 1
#endif
#if DIM > 1
		|| i.z < 2 || i.z >= SIZE_Z - 1
#endif
	) return;
	int index = INDEXV(i);
	int indexL = index;

	for (int side = 0; side < DIM; ++side) {
		int indexR = index + stepsize[side];
		
		for (int j = 0; j < NUM_STATES; ++j) {
			real fluxL = fluxBuffer[j + NUM_STATES * (side + DIM * indexL)];
			real fluxR = fluxBuffer[j + NUM_STATES * (side + DIM * indexR)];
			real deltaFlux = fluxR - fluxL;
			stateBuffer[j + NUM_STATES * index] -= deltaFlux * dt_dx[side];
		}
	}
}

