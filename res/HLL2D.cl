#include "HydroGPU/Shared/Common2D.h"

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
	const __global real* gravityPotentialBuffer,
	int2 size)
{
	int2 i = (int2)(get_global_id(0), get_global_id(1));
	if (i.x < 2 || i.y < 2 || i.x >= size.x - 1 || i.y >= size.y - 1) return;
	int index = INDEXV(i);

	for (int side = 0; side < 2; ++side) {	
		int2 iPrev = i;
		--iPrev[side];
		int indexPrev = INDEXV(iPrev);

		int interfaceIndex = side + 2 * index;

		real4 stateL = stateBuffer[indexPrev];
		real4 stateR = stateBuffer[index];

		real densityL = stateL.x;
		real invDensityL = 1.f / densityL;
		real2 velocityL = stateL.yz * invDensityL;
		real energyTotalL = stateL.w * invDensityL;

		real densityR = stateR.x;
		real invDensityR = 1.f / densityR;
		real2 velocityR = stateR.yz * invDensityR;
		real energyTotalR = stateR.w * invDensityR;

		real energyKineticL = .5f * dot(velocityL, velocityL);
		real energyPotentialL = gravityPotentialBuffer[indexPrev];
		real energyInternalL = energyTotalL - energyKineticL - energyPotentialL;
		real pressureL = (GAMMA - 1.f) * densityL * energyInternalL;
		real enthalpyTotalL = energyTotalL + pressureL * invDensityL;
		real weightL = sqrt(densityL);

		real energyKineticR = .5f * dot(velocityR, velocityR);
		real energyPotentialR = gravityPotentialBuffer[index];
		real energyInternalR = energyTotalR - energyKineticR - energyPotentialR;
		real pressureR = (GAMMA - 1.f) * densityR * energyInternalR;
		real enthalpyTotalR = energyTotalR + pressureR * invDensityR;
		real weightR = sqrt(densityR);

		real roeWeightNormalization = 1.f / (weightL + weightR);
		real2 velocity = (weightL * velocityL + weightR * velocityR) * roeWeightNormalization;
		real enthalpyTotal = (weightL * enthalpyTotalL + weightR * enthalpyTotalR) * roeWeightNormalization;
		
		real velocitySq = dot(velocity, velocity);
		real speedOfSound = sqrt((GAMMA - 1.f) * (enthalpyTotal - .5f * velocitySq));

		real2 normal = (real2)(0.f, 0.f);
		normal[side] = 1.f;
		real velocityN = dot(velocity, normal);
	
		//eigenvalues

		real4 eigenvalues = (real4)(
			velocityN - speedOfSound,
			velocityN,
			velocityN,
			velocityN + speedOfSound);
		
		eigenvaluesBuffer[interfaceIndex] = eigenvalues;
	
		//flux

		real4 fluxL = flux(densityL, velocityL, pressureL, enthalpyTotalL, normal);
		real4 fluxR = flux(densityR, velocityR, pressureR, enthalpyTotalR, normal);
	
		real sl = eigenvalues.x;
		real sr = eigenvalues.w;

		real4 flux;
		if (sl < 0.f) {
			flux = fluxL;
		} else if (sr > 0.f) {
			flux = fluxR;
		} else {
			flux = (sr * fluxL - sl * fluxR + sl * sr * (stateR - stateL)) / (sr - sl);
		}
	}
}

//do we want to use interface wavespeeds to calculate cfl?
// esp when the flux values are computed from cell wavespeeds
__kernel void calcCFL(
	__global real* cflBuffer,
	const __global real4* eigenvaluesBuffer,
	int2 size,
	real2 dx,
	real cfl)
{
	int2 i = (int2)(get_global_id(0), get_global_id(1));
	int index = INDEXV(i);
	if (i.x < 2 || i.y < 2 || i.x >= size.x - 2 || i.y >= size.y - 2) {
		cflBuffer[index] = INFINITY;
		return;
	}

	real2 dum;
	for (int side = 0; side < 2; ++side) {
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

		dum[side] = dx[side] / (maxLambda - minLambda);
	}
		
	cflBuffer[index] = cfl * min(dum.x, dum.y);
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
	int2 size,
	real2 dx,
	const __global real* dt)
{
	real2 dt_dx = *dt / dx;
	
	int2 i = (int2)(get_global_id(0), get_global_id(1));
	if (i.x < 2 || i.y < 2 || i.x >= size.x - 2 || i.y >= size.y - 2) return;
	int index = INDEXV(i);
	
	for (int side = 0; side < 2; ++side) {	
		int2 iNext = i;
		++iNext[side];
		int indexNext = INDEXV(iNext);
		
		real4 fluxL = fluxBuffer[side + 2 * index];
		real4 fluxR = fluxBuffer[side + 2 * indexNext];

		real4 df = fluxR - fluxL;
		stateBuffer[index] -= df * dt_dx[side];
	}
}

