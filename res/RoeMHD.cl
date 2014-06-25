#include "HydroGPU/Shared/Common.h"

real8 matmul(real16 ma, real16 mb, real16 mc, real16 md, real8 v);
void mat8inverse(real16* ia, real16* ib, real16* ic, real16* id, real16 ma, real16 mb, real16 mc, real16 md);
real8 slopeLimiter(real8 r);

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

void mat8inverse(
	real16* ia,
	real16* ib,
	real16* ic,
	real16* id,
	real16 ma,
	real16 mb,
	real16 mc,
	real16 md)
{
	real8 input[8] = {
		ma.s01234567,
		ma.s89ABCDEF,
		mb.s01234567,
		mb.s89ABCDEF,
		mc.s01234567,
		mc.s89ABCDEF,
		md.s01234567,
		md.s89ABCDEF
	};
	int indxc[8], indxr[8], ipiv[8];
	int i, icol = 0, irow = 0, j, k, l, ll;
	real big, dum, pivinv, temp;
	for (j = 0; j < 8; ++j) {
		ipiv[j] = 0;
	}
	for (i = 0; i < 8; ++i) {
		big = 0.f;
		for ( j = 0; j < 8; ++j) {
			if (ipiv[j] != 1) {
				for (k = 0; k < 8; ++k) {
					if (ipiv[k] == 0) {
						if (fabs(input[j][k]) >= big) {
							big = fabs(input[j][k]);
							irow = j;
							icol = k;
						}
					} else {
						if(ipiv[k] > 1) {
							return;	//singular
						}
					}
				}
			}
		}
		++ipiv[icol];
		if(irow != icol) {
			for (int k = 0; k < 8; ++k) {
				temp = input[irow][k];
				input[irow][k] = input[icol][k];
				input[icol][k] = temp;
			}
		}
		indxr[i] = irow;
		indxc[i] = icol;
		if(input[icol][icol] == 0.f) {
			return;	//singular	
		}
		pivinv = 1.f / input[icol][icol];
		input[icol][icol] = 1.f;
		for (l = 0; l < 8; ++l) {
			input[icol][l] *= pivinv;
		}
		for (ll = 0; ll < 8; ++ll) {
			if (ll != icol) {
				dum = input[ll][icol];
				input[ll][icol] = 0.f;
				for (l = 0; l < 8; ++l) {
					input[ll][l] -= input[icol][l] * dum;
				}
			}
		}
	}
	
	for (l = 8-1; l >= 0; --l) {
		if (indxr[l] != indxc[l]) {
			for (k = 0; k < 8; ++k) {
				temp = input[k][indxr[l]];
				input[k][indxr[l]] = input[k][indxc[l]];
				input[k][indxc[l]] = temp;
			}
		}
	}

	*ia = (real16)(input[0], input[1]);
	*ib = (real16)(input[2], input[3]);
	*ic = (real16)(input[4], input[5]);
	*id = (real16)(input[6], input[7]);
}

__kernel void calcEigenBasis(
	__global real8* eigenvaluesBuffer,
	__global real16* eigenvectorsBuffer,
	__global real16* eigenvectorsInverseBuffer,
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

		real densityL = stateL.s0;
		real invDensityL = 1.f / densityL;
		real4 velocityL = (real4)(stateL.s1, stateL.s2, stateL.s3, 0.f) * invDensityL;
		real energyTotalL = stateL.s4 * invDensityL;
		real energyKineticL = .5f * dot(velocityL, velocityL);
		real energyPotentialL = gravityPotentialBuffer[indexPrev];
		real energyInternalL = energyTotalL - energyKineticL - energyPotentialL;
		real pressureL = (GAMMA - 1.f) * densityL * energyInternalL;
		real enthalpyTotalL = energyTotalL + pressureL * invDensityL;
		real weightL = sqrt(densityL);
		real4 magneticFieldL = (real4)(stateL.s5, stateL.s6, stateL.s7, 0.f);

		real densityR = stateR.s0;
		real invDensityR = 1.f / densityR;
		real4 velocityR = (real4)(stateR.s1, stateR.s2, stateR.s3, 0.f) * invDensityR;
		real energyTotalR = stateR.s4 * invDensityR;
		real energyKineticR = .5f * dot(velocityR, velocityR);
		real energyPotentialR = gravityPotentialBuffer[index];
		real energyInternalR = energyTotalR - energyKineticR - energyPotentialR;
		real pressureR = (GAMMA - 1.f) * densityR * energyInternalR;
		real enthalpyTotalR = energyTotalR + pressureR * invDensityR;
		real weightR = sqrt(densityR);
		real4 magneticFieldR = (real4)(stateR.s5, stateR.s6, stateR.s7, 0.f);

		real roeWeightNormalization = 1.f / (weightL + weightR);
		real density = weightL * weightR;
		real4 velocity = (weightL * velocityL + weightR * velocityR) * roeWeightNormalization;
		real velocitySq = dot(velocity, velocity);
		real enthalpyTotal = (weightL * enthalpyTotalL + weightR * enthalpyTotalR) * roeWeightNormalization;
		real speedOfSound = sqrt((GAMMA - 1.f) * (enthalpyTotal - .5f * velocitySq));
		real speedOfSoundSq = speedOfSound * speedOfSound;
		real4 magneticField = (weightL * magneticFieldL + weightR * magneticFieldR) * roeWeightNormalization;
		real magneticFieldSq = dot(magneticField, magneticField);

		real4 normal = (real4)(0.f, 0.f, 0.f, 0.f);
		normal[side] = 1;

		real magneticFieldN = dot(magneticField, normal);
		real AlfvenSpeed = magneticFieldN / density;
		real AlfvenSpeedSq = AlfvenSpeed * AlfvenSpeed;
		real starSpeedSq = .5f * magneticFieldSq / density + speedOfSound * speedOfSound;
		real starSpeedToTheFourth = starSpeedSq * starSpeedSq;
		real discriminant = sqrt(starSpeedToTheFourth - speedOfSoundSq * AlfvenSpeedSq);
		real slowSpeedSq = starSpeedSq - discriminant;
		real slowSpeed = sqrt(slowSpeedSq);
		real fastSpeedSq = starSpeedSq + discriminant;
		real fastSpeed = sqrt(fastSpeedSq);

		real4 tangentA = (real4)(normal.y, normal.z, normal.x, 0.f);
		real4 tangentB = (real4)(normal.z, normal.x, normal.y, 0.f);
		real velocityN = dot(velocity, normal);
		real velocityTA = dot(velocity, tangentA);
		real velocityTB = dot(velocity, tangentB);
	
		//eigenvalues

		real8 eigenvalues;
		eigenvalues.s0 = velocityN - fastSpeed;
		eigenvalues.s1 = velocityN - AlfvenSpeed;
		eigenvalues.s2 = velocityN - slowSpeed;
		eigenvalues.s3 = velocityN;
		eigenvalues.s4 = velocityN;
		eigenvalues.s5 = velocityN + slowSpeed;
		eigenvalues.s6 = velocityN + AlfvenSpeed;
		eigenvalues.s7 = velocityN + fastSpeed;
		eigenvaluesBuffer[interfaceIndex] = eigenvalues;

		//eigenvectors

		//specify transposed
		real16 eigenvectorsA;
		real16 eigenvectorsB;
		real16 eigenvectorsC;
		real16 eigenvectorsD;
	
		real sqrtDensity = sqrt(density);
		real magneticFieldNSign = step(0.f, magneticFieldN) * 2.f - 1.f;
		real4 magneticFieldCrossN = cross(magneticField, normal);
		//col 
		eigenvectorsA.s0 = (density * fastSpeedSq - magneticFieldSq) / speedOfSoundSq;
		eigenvectorsA.s8 = -fastSpeed * normal.x + magneticField.x * magneticFieldN / (density * fastSpeed);
		eigenvectorsB.s0 = -fastSpeed * normal.y + magneticField.y * magneticFieldN / (density * fastSpeed);
		eigenvectorsB.s8 = -fastSpeed * normal.z + magneticField.z * magneticFieldN / (density * fastSpeed);
		eigenvectorsC.s0 = magneticField.x - normal.x * magneticFieldN;
		eigenvectorsC.s8 = magneticField.y - normal.y * magneticFieldN;
		eigenvectorsD.s0 = magneticField.z - normal.z * magneticFieldN;
		eigenvectorsD.s8 = density * fastSpeedSq - magneticFieldSq;
		//col
		eigenvectorsA.s1 = 0.f;
		eigenvectorsA.s9 = magneticFieldCrossN.x * magneticFieldNSign;
		eigenvectorsB.s1 = magneticFieldCrossN.y * magneticFieldNSign;
		eigenvectorsB.s9 = magneticFieldCrossN.z * magneticFieldNSign;
		eigenvectorsC.s1 = magneticFieldCrossN.x * sqrtDensity;
		eigenvectorsC.s9 = magneticFieldCrossN.y * sqrtDensity;
		eigenvectorsD.s1 = magneticFieldCrossN.z * sqrtDensity;
		eigenvectorsD.s9 = 0.f;
		//col
		eigenvectorsA.s2 = (density * slowSpeedSq - magneticFieldSq) / speedOfSoundSq;
		eigenvectorsA.sA = -normal.x * slowSpeed + magneticField.x * magneticFieldN / (density * slowSpeed);
		eigenvectorsB.s2 = -normal.y * slowSpeed + magneticField.y * magneticFieldN / (density * slowSpeed);
		eigenvectorsB.sA = -normal.z * slowSpeed + magneticField.z * magneticFieldN / (density * slowSpeed);
		eigenvectorsC.s2 = magneticField.x - normal.x * magneticFieldN;
		eigenvectorsC.sA = magneticField.y - normal.y * magneticFieldN;
		eigenvectorsD.s2 = magneticField.z - normal.z * magneticFieldN;
		eigenvectorsD.sA = density * slowSpeedSq - magneticFieldSq;
		//col
		eigenvectorsA.s3 = 1.f;
		eigenvectorsA.sB = 0.f; 
		eigenvectorsB.s3 = 0.f;
		eigenvectorsB.sB = 0.f;
		eigenvectorsC.s3 = 0.f;
		eigenvectorsC.sB = 0.f;
		eigenvectorsD.s3 = 0.f;
		eigenvectorsD.sB = 0.f;	
		//col 
		eigenvectorsA.s4 = 0.f;
		eigenvectorsA.sC = 0.f;
		eigenvectorsB.s4 = 0.f;
		eigenvectorsB.sC = 0.f;
		eigenvectorsC.s4 = normal.x;
		eigenvectorsC.sC = normal.y;
		eigenvectorsD.s4 = normal.z;
		eigenvectorsD.sC = 0.f;
		//col
		eigenvectorsA.s5 = (density * slowSpeedSq - magneticFieldSq) / speedOfSoundSq;
		eigenvectorsA.sD = normal.x * slowSpeed - magneticField.x * magneticFieldN / (density * slowSpeed);
		eigenvectorsB.s5 = normal.y * slowSpeed - magneticField.y * magneticFieldN / (density * slowSpeed);
		eigenvectorsB.sD = normal.z * slowSpeed - magneticField.z * magneticFieldN / (density * slowSpeed);
		eigenvectorsC.s5 = magneticField.x - normal.x * magneticFieldN;
		eigenvectorsC.sD = magneticField.y - normal.y * magneticFieldN;
		eigenvectorsD.s5 = magneticField.z - normal.z * magneticFieldN;
		eigenvectorsD.sD = density * slowSpeedSq - magneticFieldSq;
		//col
		eigenvectorsA.s6 = 0.f;
		eigenvectorsA.sE = -magneticFieldCrossN.x * magneticFieldNSign;
		eigenvectorsB.s6 = -magneticFieldCrossN.y * magneticFieldNSign;
		eigenvectorsB.sE = -magneticFieldCrossN.z * magneticFieldNSign;
		eigenvectorsC.s6 = magneticFieldCrossN.x * sqrtDensity;
		eigenvectorsC.sE = magneticFieldCrossN.y * sqrtDensity;
		eigenvectorsD.s6 = magneticFieldCrossN.z * sqrtDensity;
		eigenvectorsD.sE = 0.f;
		//col
		eigenvectorsA.s7 = (density * fastSpeedSq - magneticFieldSq) / speedOfSoundSq;
		eigenvectorsA.sF = normal.x * fastSpeed - magneticField.x * magneticFieldN / (density * fastSpeed);
		eigenvectorsB.s7 = normal.y * fastSpeed - magneticField.y * magneticFieldN / (density * fastSpeed);
		eigenvectorsB.sF = normal.z * fastSpeed - magneticField.z * magneticFieldN / (density * fastSpeed);
		eigenvectorsC.s7 = magneticField.x - normal.x * magneticFieldN;
		eigenvectorsC.sF = magneticField.y - normal.y * magneticFieldN;
		eigenvectorsD.s7 = magneticField.z - normal.z * magneticFieldN;
		eigenvectorsD.sF = density * fastSpeedSq - magneticFieldSq;
		
		eigenvectorsBuffer[0 + 4 * interfaceIndex] = eigenvectorsA;
		eigenvectorsBuffer[1 + 4 * interfaceIndex] = eigenvectorsB;
		eigenvectorsBuffer[2 + 4 * interfaceIndex] = eigenvectorsC;
		eigenvectorsBuffer[3 + 4 * interfaceIndex] = eigenvectorsD;

		//eigenvectors inverse

		real16 eigenvectorsInverseA;
		real16 eigenvectorsInverseB;
		real16 eigenvectorsInverseC;
		real16 eigenvectorsInverseD;
		mat8inverse(
			&eigenvectorsInverseA,
			&eigenvectorsInverseB,
			&eigenvectorsInverseC,
			&eigenvectorsInverseD,
			eigenvectorsA,
			eigenvectorsB,
			eigenvectorsC,
			eigenvectorsD);
			
		eigenvectorsInverseBuffer[0 + 4 * interfaceIndex] = eigenvectorsInverseA;
		eigenvectorsInverseBuffer[1 + 4 * interfaceIndex] = eigenvectorsInverseB;
		eigenvectorsInverseBuffer[2 + 4 * interfaceIndex] = eigenvectorsInverseC;
		eigenvectorsInverseBuffer[3 + 4 * interfaceIndex] = eigenvectorsInverseD;
	}
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

real8 slopeLimiter(real8 r) {
	//donor cell
	//return (real8)(0.f);
	//superbee
	return max(0.f, max(min(1.f, 2.f * r), min(2.f, r)));
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

