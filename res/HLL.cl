#include "HydroGPU/Shared/Common.h"

real4 matmul(real16 m, real4 v);
real4 flux(real density, real2 velocity, real pressure, real enthalpyTotal, real2 normal);
real4 slopeLimiter(real4 r);

real4 matmul(real16 m, real4 v) {
	return (real4)(
		dot(m.s0123, v),
		dot(m.s4567, v),
		dot(m.s89AB, v),
		dot(m.sCDEF, v));
}

real4 flux(real density, real2 velocity, real pressure, real enthalpyTotal, real2 normal) {
	real velocityN = dot(velocity, normal);
	return (real4)(
		density * enthalpyTotal * velocityN,
		density * velocity.x * velocityN + pressure * normal.x,
		density * velocity.y * velocityN + pressure * normal.y,
		density * velocityN);
}

__kernel void calcEigenvalues(
	__global real4* eigenvaluesBuffer,
	__global real4* fluxBuffer,
	const __global real4* stateBuffer,
	const __global real* gravityPotentialBuffer)
{
	int2 i = (int2)(get_global_id(0), get_global_id(1));
	if (i.x < 2 || i.x >= SIZE_X - 1 
#if DIM > 1
		|| i.y < 2 || i.y >= SIZE_Y - 1
#endif
	) return;
	int index = INDEXV(i);

	for (int side = 0; side < DIM; ++side) {	
		int2 iPrev = i;
		--iPrev[side];
		int indexPrev = INDEXV(iPrev);
		
		real2 normal = (real2)(0.f, 0.f);
		normal[side] = 1.f;

		int interfaceIndex = side + 2 * index;

		real4 stateL = stateBuffer[indexPrev];
		real4 stateR = stateBuffer[index];

		real densityL = stateL.x;
		real invDensityL = 1.f / densityL;
		real2 velocityL = stateL.yz * invDensityL;
		real velocityNL = dot(velocityL, normal);
		real velocitySqL = dot(velocityL, velocityL);
		real energyTotalL = stateL.w * invDensityL;
		real energyKineticL = .5f * dot(velocityL, velocityL);
		real energyPotentialL = gravityPotentialBuffer[indexPrev];
		real energyInternalL = energyTotalL - energyKineticL - energyPotentialL;
		real pressureL = (GAMMA - 1.f) * densityL * energyInternalL;
		real enthalpyTotalL = energyTotalL + pressureL * invDensityL;
		real speedOfSoundL = sqrt((GAMMA - 1.f) * (enthalpyTotalL - .5f * velocitySqL));
		real4 eigenvaluesL = (real4)(
			velocityNL - speedOfSoundL,
			velocityNL,
			velocityNL,
			velocityNL + speedOfSoundL);
		real weightL = sqrt(densityL);

		real densityR = stateR.x;
		real invDensityR = 1.f / densityR;
		real2 velocityR = stateR.yz * invDensityR;
		real velocityNR = dot(velocityR, normal);
		real velocitySqR = dot(velocityR, velocityR);
		real energyTotalR = stateR.w * invDensityR;
		real energyKineticR = .5f * dot(velocityR, velocityR);
		real energyPotentialR = gravityPotentialBuffer[index];
		real energyInternalR = energyTotalR - energyKineticR - energyPotentialR;
		real pressureR = (GAMMA - 1.f) * densityR * energyInternalR;
		real enthalpyTotalR = energyTotalR + pressureR * invDensityR;
		real speedOfSoundR = sqrt((GAMMA - 1.f) * (enthalpyTotalR - .5f * velocitySqR));
		real4 eigenvaluesR = (real4)(
			velocityNR - speedOfSoundR,
			velocityNR,
			velocityNR,
			velocityNR + speedOfSoundR);
		real weightR = sqrt(densityR);

		//Roe averaging
		real roeWeightNormalization = 1.f / (weightL + weightR);
		real2 velocity = (weightL * velocityL + weightR * velocityR) * roeWeightNormalization;
		real velocityN = dot(velocity, normal);
		real velocitySq = dot(velocity, velocity);
		real enthalpyTotal = (weightL * enthalpyTotalL + weightR * enthalpyTotalR) * roeWeightNormalization;
		real speedOfSound = sqrt((GAMMA - 1.f) * (enthalpyTotal - .5f * velocitySq));
		real4 eigenvaluesRoe = (real4)(
			velocityN - speedOfSound,
			velocityN,
			velocityN,
			velocityN + speedOfSound);
		
		eigenvaluesBuffer[interfaceIndex] = eigenvaluesRoe;
	
		//flux

		real4 fluxL = flux(densityL, velocityL, pressureL, enthalpyTotalL, normal);
		real4 fluxR = flux(densityR, velocityR, pressureR, enthalpyTotalR, normal);
	
		real sl = min(eigenvaluesL.s0, eigenvaluesRoe.s0);
		real sr = max(eigenvaluesR.s3, eigenvaluesRoe.s3);

		real4 flux;
		if (sl < 0.f) {
			flux = fluxL;
		} else if (sr > 0.f) {
			flux = fluxR;
		} else {
			flux = (sr * fluxL - sl * fluxR + sl * sr * (stateR - stateL)) / (sr - sl);
		}

		fluxBuffer[interfaceIndex] = flux;
	}
}

//do we want to use interface wavespeeds to calculate cfl?
// esp when the flux values are computed from cell wavespeeds
__kernel void calcCFL(
	__global real* cflBuffer,
	const __global real4* eigenvaluesBuffer,
	real2 dx,
	real cfl)
{
	int2 i = (int2)(get_global_id(0), get_global_id(1));
	int index = INDEXV(i);
	if (i.x < 2 || i.x >= SIZE_X - 2 
#if DIM > 1
		|| i.y < 2 || i.y >= SIZE_Y - 2
#endif
	) {
		cflBuffer[index] = INFINITY;
		return;
	}

	real result = INFINITY;
	for (int side = 0; side < DIM; ++side) {
		int2 iNext = i;
		++iNext[side];
		int indexNext = INDEXV(iNext);
		
		real4 eigenvaluesL = eigenvaluesBuffer[side + 2 * index];
		real4 eigenvaluesR = eigenvaluesBuffer[side + 2 * indexNext];
		
		real maxLambda = max(
			0.f,
			max(
				max(
					eigenvaluesL.x,
					eigenvaluesL.y), 
				max(
					eigenvaluesL.z,
					eigenvaluesL.w)));

		real minLambda = min(
			0.f,
			min(
				min(
					eigenvaluesR.x,
					eigenvaluesR.y),
				min(
					eigenvaluesR.z,
					eigenvaluesR.w)));

		real dum = dx[side] / (maxLambda - minLambda);
		result = min(result, dum);
	}
		
	cflBuffer[index] = cfl * result;
}

real4 slopeLimiter(real4 r) {
	//donor cell
	//return (real4)(0.f, 0.f, 0.f, 0.f);
	//superbee
	return max(0.f, max(min(1.f, 2.f * r), min(2.f, r)));
}

__kernel void integrateFlux(
	__global real4* stateBuffer,
	const __global real4* fluxBuffer,
	real2 dx,
	const __global real* dt)
{
	real2 dt_dx = *dt / dx;
	
	int2 i = (int2)(get_global_id(0), get_global_id(1));
	if (i.x < 2 || i.x >= SIZE_X - 2 
#if DIM > 1
		|| i.y < 2 || i.y >= SIZE_Y - 2
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

