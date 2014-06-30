#include "HydroGPU/Shared/Common.h"

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
	
		__global real* eigenvalues = eigenvaluesBuffer + NUM_STATES * interfaceIndex;
		__global real* flux = fluxBuffer + NUM_STATES * interfaceIndex;

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
		real roeWeightL = sqrt(densityL);
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
		real roeWeightR = sqrt(densityR);
		real speedOfSoundR = sqrt((GAMMA - 1.f) * (enthalpyTotalR - .5f * velocitySqR));
		real eigenvaluesMaxR = velocityNR + speedOfSoundR;
		
		real roeWeightNormalization = 1.f / (roeWeightL + roeWeightR);
		real4 velocity = (roeWeightL * velocityL + roeWeightR * velocityR) * roeWeightNormalization;
		real enthalpyTotal = (roeWeightL * enthalpyTotalL + roeWeightR * enthalpyTotalR) * roeWeightNormalization;
		
		real velocitySq = dot(velocity, velocity);
		real speedOfSound = sqrt((GAMMA - 1.f) * (enthalpyTotal - .5f * velocitySq));
		real velocityN = dot(velocity, normal);
	
		//eigenvalues

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

#if 0	//Davis direct
		real sl = eigenvaluesMinL;
		real sr = eigenvaluesMaxR;
#endif
#if 1	//Davis direct bounded
		real sl = min(eigenvaluesMinL, eigenvalues[0]);
		real sr = max(eigenvaluesMaxR, eigenvalues[NUM_STATES-1]);
#endif
		
		if (sl >= 0.f) {
			//fluxL
			flux[0] = densityL * velocityNL;
			flux[1] = densityL * velocityL.x * velocityNL + normal.x * pressureL;
#if DIM > 1
			flux[2] = densityL * velocityL.y * velocityNL + normal.y * pressureL;
#endif
#if DIM > 2
			flux[3] = densityL * velocityL.z * velocityNL + normal.z * pressureL;
#endif
			flux[DIM+1] = densityL * enthalpyTotalL * velocityNL;
		} else if (sl <= 0.f && sr >= 0.f) {
			//(sr * fluxL[j] - sl * fluxR[j] + sl * sr * (stateR[j] - stateL[j])) / (sr - sl)
			real invDenom = 1.f / (sr - sl);
			flux[0] = (sr * densityL * velocityNL
					- sl * densityR * velocityNR
					+ sl * sr * (densityR - densityL)) * invDenom;
			flux[1] = (sr * (densityL * velocityL.x * velocityNL + normal.x * pressureL)
					- sl * (densityR * velocityR.x * velocityNR + normal.x * pressureR)
					+ sl * sr * (densityR * velocityR.x - densityL * velocityL.x)) * invDenom;
#if DIM > 1
			flux[2] = (sr * (densityL * velocityL.y * velocityNL + normal.y * pressureL)
					- sl * (densityR * velocityR.y * velocityNR + normal.y * pressureR)
					+ sl * sr * (densityR * velocityR.y - densityL * velocityL.y)) * invDenom;
#endif
#if DIM > 2
			flux[3] = (sr * (densityL * velocityL.z * velocityNL + normal.z * pressureL)
					- sl * (densityR * velocityR.z * velocityNR + normal.z * pressureR)
					+ sl * sr * (densityR * velocityR.z - densityL * velocityL.z)) * invDenom;
#endif
			flux[DIM+1] = (sr * (densityL * enthalpyTotalL * velocityNL)
						- sl * (densityR * enthalpyTotalR * velocityNR)
						+ sl * sr * (densityR * energyTotalR - densityL * energyTotalL)) * invDenom;
		} else if (sr <= 0.f) {
			//fluxR
			flux[0] = densityR * velocityNR;
			flux[1] = densityR * velocityR.x * velocityNR + normal.x * pressureR;
#if DIM > 1
			flux[2] = densityR * velocityR.y * velocityNR + normal.y * pressureR;
#endif
#if DIM > 2
			flux[3] = densityR * velocityR.z * velocityNR + normal.z * pressureR;
#endif
			flux[DIM+1] = densityR * enthalpyTotalR * velocityNR;	
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
#if DIM > 2
		|| i.z < 2 || i.z >= SIZE_Z - 1
#endif
	) return;
	int index = INDEXV(i);
	int indexL = index;

	for (int side = 0; side < DIM; ++side) {
		int indexR = index + stepsize[side];
		const __global real* fluxL = fluxBuffer + NUM_STATES * (side + DIM * indexL);
		const __global real* fluxR = fluxBuffer + NUM_STATES * (side + DIM * indexR);
		__global real* state = stateBuffer + NUM_STATES * index;
		for (int j = 0; j < NUM_STATES; ++j) {
			real deltaFlux = fluxR[j] - fluxL[j];
			state[j] -= deltaFlux * dt_dx[side];
		}
	}
}

