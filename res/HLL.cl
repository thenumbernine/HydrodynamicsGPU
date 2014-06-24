#include "HydroGPU/Shared/Common.h"

real8 flux(real density, real4 velocity, real pressure, real enthalpyTotal, real4 normal);
real8 slopeLimiter(real8 r);

real8 flux(real density, real4 velocity, real pressure, real enthalpyTotal, real4 normal) {
	real velocityN = dot(velocity, normal);
	return (real8)(
		density * velocityN,
		density * velocity.x * velocityN + pressure * normal.x,
		density * velocity.y * velocityN + pressure * normal.y,
		density * velocity.z * velocityN + pressure * normal.z,
		density * enthalpyTotal * velocityN,
		0.f,
		0.f,
		0.f);
}

real8 slopeLimiter(real8 r) {
	//donor cell
	//return 0.f;
	//superbee
	return max(0.f, max(min(1.f, 2.f * r), min(2.f, r)));
}

__kernel void calcFluxAndEigenvalues(
	__global real8* eigenvaluesBuffer,
	__global real8* fluxBuffer,
	const __global real8* stateBuffer,
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

	for (int side = 0; side < DIM; ++side) {	
		int4 iPrev = i;
		--iPrev[side];
		int indexPrev = INDEXV(iPrev);
		
		int interfaceIndex = side + DIM * index;

		real8 stateL = stateBuffer[indexPrev];
		real8 stateR = stateBuffer[index];
		
		real4 normal = (real4)(0.f, 0.f, 0.f, 0.f);
		normal[side] = 1;

		real densityL = stateL.s0;
		real invDensityL = 1.f / densityL;
		real4 velocityL = (real4)(stateL.s1, stateL.s2, stateL.s3, 0.f) * invDensityL;
		real velocitySqL = dot(velocityL, velocityL);
		real velocityNL = dot(velocityL, normal);
		real energyTotalL = stateL.s4 * invDensityL;
		real energyKineticL = .5f * dot(velocityL, velocityL);
		real energyPotentialL = gravityPotentialBuffer[indexPrev];
		real energyInternalL = energyTotalL - energyKineticL - energyPotentialL;
		real pressureL = (GAMMA - 1.f) * densityL * energyInternalL;
		real enthalpyTotalL = energyTotalL + pressureL * invDensityL;
		real weightL = sqrt(densityL);
		real speedOfSoundL = sqrt((GAMMA - 1.f) * (enthalpyTotalL - .5f * velocitySqL));
		real8 eigenvaluesL = (real8)(
			velocityNL - speedOfSoundL,
			velocityNL,
			velocityNL,
			velocityNL,
			velocityNL + speedOfSoundL,
			0.f,
			0.f,
			0.f);

		real densityR = stateR.s0;
		real invDensityR = 1.f / densityR;
		real4 velocityR = (real4)(stateR.s1, stateR.s2, stateR.s3, 0.f) * invDensityR;
		real velocitySqR = dot(velocityR, velocityR);
		real velocityNR = dot(velocityR, normal);
		real energyTotalR = stateR.s4 * invDensityR;
		real energyKineticR = .5f * dot(velocityR, velocityR);
		real energyPotentialR = gravityPotentialBuffer[index];
		real energyInternalR = energyTotalR - energyKineticR - energyPotentialR;
		real pressureR = (GAMMA - 1.f) * densityR * energyInternalR;
		real enthalpyTotalR = energyTotalR + pressureR * invDensityR;
		real weightR = sqrt(densityR);
		real speedOfSoundR = sqrt((GAMMA - 1.f) * (enthalpyTotalR - .5f * velocitySqR));
		real8 eigenvaluesR = (real8)(
			velocityNR - speedOfSoundR,
			velocityNR,
			velocityNR,
			velocityNR,
			velocityNR + speedOfSoundR,
			0.f,
			0.f,
			0.f);
		
		real roeWeightNormalization = 1.f / (weightL + weightR);
		real4 velocity = (weightL * velocityL + weightR * velocityR) * roeWeightNormalization;
		real enthalpyTotal = (weightL * enthalpyTotalL + weightR * enthalpyTotalR) * roeWeightNormalization;
		
		real velocitySq = dot(velocity, velocity);
		real speedOfSound = sqrt((GAMMA - 1.f) * (enthalpyTotal - .5f * velocitySq));
		real velocityN = dot(velocity, normal);
	
		//eigenvalues
		
		real8 eigenvalues;
		eigenvalues.s0 = velocityN - speedOfSound;
		eigenvalues.s1 = velocityN;
		eigenvalues.s2 = velocityN;
		eigenvalues.s3 = velocityN;
		eigenvalues.s4 = velocityN + speedOfSound;
		eigenvalues.s5 = 0.f;
		eigenvalues.s6 = 0.f;
		eigenvalues.s7 = 0.f;
		eigenvaluesBuffer[interfaceIndex] = eigenvalues;

		//flux

		real8 fluxL = flux(densityL, velocityL, pressureL, enthalpyTotalL, normal);
		real8 fluxR = flux(densityR, velocityR, pressureR, enthalpyTotalR, normal);

#if 0	//Davis direct
		real sl = eigenvaluesL.s0;
		real sr = eigenvaluesR.s4;
#endif
#if 1	//Davis direct bounded
		real sl = min(eigenvaluesL.s0, eigenvalues.s0);
		real sr = max(eigenvaluesR.s4, eigenvalues.s4);
#endif
		
		real8 flux;
		if (sl >= 0.f) {
			flux = fluxL;
		} else if (sl <= 0.f && sr >= 0.f) {
			flux = (sr * fluxL - sl * fluxR + sl * sr * (stateR - stateL)) / (sr - sl);
		} else if (sr <= 0.f) {
			flux = fluxR;
		}

		fluxBuffer[interfaceIndex] = flux;
	}
}

//do we want to use interface wavespeeds to calculate cfl?
// esp when the flux values are computed from cell wavespeeds
__kernel void calcCFL(
	__global real* cflBuffer,
	const __global real8* eigenvaluesBuffer,
	real cfl)
{
	real4 dx = (real4)(DX, DY, DZ, 1.f);
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

	real result = INFINITY;
	for (int side = 0; side < DIM; ++side) {
		int4 iNext = i;
		++iNext[side];
		int indexNext = INDEXV(iNext);
		
		real8 eigenvaluesL = eigenvaluesBuffer[side + DIM * index];
		real8 eigenvaluesR = eigenvaluesBuffer[side + DIM * indexNext];
		
		real maxLambda = max(
			max(
				0.f,
				eigenvaluesL.s0),
			max(
				max(
					eigenvaluesL.s1,
					eigenvaluesL.s2), 
				max(
					eigenvaluesL.s3,
					eigenvaluesL.s4)));

		real minLambda = min(
			min(
				0.f,
				eigenvaluesR.s0),
			min(
				min(
					eigenvaluesR.s1,
					eigenvaluesR.s2),
				min(
					eigenvaluesR.s3,
					eigenvaluesR.s4)));

		real dum = dx[side] / (maxLambda - minLambda);
		result = min(result, dum);
	}
		
	cflBuffer[index] = cfl * result;
}

__kernel void integrateFlux(
	__global real8* stateBuffer,
	const __global real8* fluxBuffer,
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

