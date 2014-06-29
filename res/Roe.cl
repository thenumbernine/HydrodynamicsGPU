#include "HydroGPU/Shared/Common.h"

real8 matmul(real16 ma, real16 mb, real16 mc, real16 md, real8 v);

real8 matmul(real16 ma, real16 mb, real16 mc, real16 md, real8 v) {
	//why do the specs say no dot product exists for float8, but with double or half extension you get dot for double8 or half8?
	return (real8)(
		dot(ma.s0123, v.s0123) + dot(ma.s4567, v.s4567),
		dot(ma.s89AB, v.s0123) + dot(ma.sCDEF, v.s4567),
		dot(mb.s0123, v.s0123) + dot(mb.s4567, v.s4567),
		dot(mb.s89AB, v.s0123) + dot(mb.sCDEF, v.s4567),
		dot(mc.s0123, v.s0123) + dot(mc.s4567, v.s4567),
		dot(mc.s89AB, v.s0123) + dot(mc.sCDEF, v.s4567),
		dot(md.s0123, v.s0123) + dot(md.s4567, v.s4567),
		dot(md.s89AB, v.s0123) + dot(md.sCDEF, v.s4567));
}

__kernel void calcCFL(
	__global real* cflBuffer,
	const __global real8* eigenvaluesBuffer,
	real cfl)
{
	real4 dx = (real4)(DX, DY, DZ, 1.f);
	int4 i = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);
	int index = INDEXV(i);
	if (i.x < 2 || i.x >= SIZE_X - 2 
#if DIM > 1
		|| i.y < 2 || i.y >= SIZE_Y - 2
#endif
#if DIM > 2
		|| i.z < 2 || i.z >= SIZE_Z - 2
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

__kernel void calcDeltaQTilde(
	__global real8* deltaQTildeBuffer,
	const __global real16* eigenvectorsInverseBuffer,
	const __global real8* stateBuffer)
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
				
		real8 stateL = stateBuffer[indexPrev];
		real8 stateR = stateBuffer[index];

		int interfaceIndex = side + DIM * index;
		
		real8 deltaQ = stateR - stateL;
		deltaQTildeBuffer[interfaceIndex] = matmul(
			eigenvectorsInverseBuffer[0 + 4 * interfaceIndex], 
			eigenvectorsInverseBuffer[1 + 4 * interfaceIndex], 
			eigenvectorsInverseBuffer[2 + 4 * interfaceIndex], 
			eigenvectorsInverseBuffer[3 + 4 * interfaceIndex], 
			deltaQ);
	}
}

#define ONLY_WORKING_IN_1D

__kernel void calcFlux(
	__global real8* fluxBuffer,
	const __global real8* stateBuffer,
	const __global real8* eigenvaluesBuffer,
	const __global real16* eigenvectorsBuffer,
	const __global real16* eigenvectorsInverseBuffer,
	const __global real8* deltaQTildeBuffer,
	const __global real* dtBuffer)
{
#ifdef ONLY_WORKING_IN_1D
	float dt = dtBuffer[0];
	real4 dt_dx = (real4)(dt / DX, dt / DY, dt / DZ, dt);
#endif

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

		int4 iNext = i;
		++iNext[side];
		int indexNext = INDEXV(iNext);
	
		real8 stateL = stateBuffer[indexPrev];
		real8 stateR = stateBuffer[index];

#ifdef ONLY_WORKING_IN_1D
		int interfaceLIndex = side + DIM * indexPrev;
#endif
		int interfaceIndex = side + DIM * index;
#ifdef ONLY_WORKING_IN_1D
		int interfaceRIndex = side + DIM * indexNext;
#endif

#ifdef ONLY_WORKING_IN_1D
		real8 deltaQTildeL = deltaQTildeBuffer[interfaceLIndex];
#endif
		real8 deltaQTilde = deltaQTildeBuffer[interfaceIndex];
#ifdef ONLY_WORKING_IN_1D
		real8 deltaQTildeR = deltaQTildeBuffer[interfaceRIndex];
#endif

		real8 eigenvalues = eigenvaluesBuffer[interfaceIndex];

		//with step function...
		real8 theta = step(0.f, eigenvalues) * 2.f - 1.f;
		//without step function...
//		real8 theta;
//		for (int i = 0; i < 8; ++i) {
//			if (eigenvalues[i] >= 0.f) {
//				theta[i] = 1.f;
//			} else {
//				theta[i] = -1.f;
//			}
//		}
	
		//this operation is crashing on compile for some cases
#ifdef ONLY_WORKING_IN_1D
		real8 rTilde = mix(deltaQTildeR, deltaQTildeL, theta * .5f + .5f) / deltaQTilde;
#endif
		//here it is written out as conditions
//		real8 rTilde;
//		for (int i = 0; i < 8; ++i) {
//			if (eigenvalues[i] >= 0.f) {
//				rTilde[i] = deltaQTildeL[i];
//			} else {
//				rTilde[i] = deltaQTildeR[i];
//			}
//			rTilde[i] /= deltaQTilde[i];
//		}
		
		real8 qAvg = (stateR + stateL) * .5f;
		real8 fluxAvgTilde = matmul(
			eigenvectorsInverseBuffer[0 + 4 * interfaceIndex], 
			eigenvectorsInverseBuffer[1 + 4 * interfaceIndex], 
			eigenvectorsInverseBuffer[2 + 4 * interfaceIndex], 
			eigenvectorsInverseBuffer[3 + 4 * interfaceIndex], 
			qAvg) * eigenvalues;

#ifdef ONLY_WORKING_IN_1D
		real8 phi = slopeLimiter(rTilde);
		real8 epsilon = eigenvalues * dt_dx[side];	//enabling this causes us to crash
#endif

		real8 fluxTilde = fluxAvgTilde; 
	
		real8 deltaFluxTilde = eigenvalues * deltaQTilde;
		
		//with theta...
		fluxTilde -= .5f * deltaFluxTilde * theta;
		//without theta... this is crashing
		//for (int i = 0; i < 8; ++i) {
		//	if (eigenvalues[i] >= 0.f) {
		//		fluxTilde[i] -= .5f * deltaFluxTilde[i];
		//	} else {
		//		fluxTilde[i] += .5f * deltaFluxTilde[i];
		//	}
		//}
	
		//with theta
		fluxTilde -= .5f * deltaFluxTilde * phi * (epsilon - theta) / (float)DIM;
		//without theta
		//for (int i = 0; i < 8; ++i) {
		//	if (eigenvalues[i] >= 0.f) {
		//		fluxTilde[i] -= .5f * .5f * deltaFluxTilde[i] * phi[i] * (epsilon[i] - 1.f);
		//	} else {
		//		fluxTilde[i] -= .5f * .5f * deltaFluxTilde[i] * phi[i] * (epsilon[i] + 1.f);
		//	}
		//}

		//combining both condition loops without theta
		//this is crashing even without references to dtBuffer
		//real dt_dx_side = 0.f;//dt_dx[side];
		//for (int i = 0; i < 4; ++i) {
		//	if (eigenvalues[i] >= 0.f) {
		//		fluxTilde[i] -= deltaFluxTilde[i] * .5f;// * (1.f - .5f * phi[i] * (1.f - eigenvalues[i] * dt_dx_side));
		//	} else {
		//		fluxTilde[i] += deltaFluxTilde[i] * .5f;// * (1.f - .5f * phi[i] * (1.f + eigenvalues[i] * dt_dx_side));
		//	}
		//}
		
		fluxBuffer[side + DIM * index] = matmul(
			eigenvectorsBuffer[0 + 4 * interfaceIndex],
			eigenvectorsBuffer[1 + 4 * interfaceIndex],
			eigenvectorsBuffer[2 + 4 * interfaceIndex],
			eigenvectorsBuffer[3 + 4 * interfaceIndex],
			fluxTilde);
	}
}

__kernel void integrateFlux(
	__global real8* stateBuffer,
	const __global real8* fluxBuffer,
	const __global real* dtBuffer)
{
	real4 dx = (real4)(DX, DY, DZ, 1.f);
	real dt = dtBuffer[0];
	real4 dt_dx = dt / dx;
	
	int4 i = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);
	if (i.x < 2 || i.x >= SIZE_X - 2 
#if DIM > 1
		|| i.y < 2 || i.y >= SIZE_Y - 2
#endif
#if DIM > 2
		|| i.z < 2 || i.z >= SIZE_Z - 2
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

