/*
The components of the Roe solver specific to the MHD equations
paritcularly the spectral decomposition
*/

#include "HydroGPU/Shared/Common.h"

void calcEigenBasisSide(
	__global real* eigenvaluesBuffer,
	__global real* eigenvectorsBuffer,
	__global real* eigenvectorsInverseBuffer,
	const __global real* stateBuffer,
	const __global real* gravityPotentialBuffer,
	int side);

void calcEigenBasisSide(
	__global real* eigenvaluesBuffer,
	__global real* eigenvectorsBuffer,
	__global real* eigenvectorsInverseBuffer,
	const __global real* stateBuffer,
	const __global real* gravityPotentialBuffer,
	int side)
{
	int4 i = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);
	
	int index = INDEXV(i);
	int indexPrev = INDEXV(iPrev);
	int interfaceIndex = side + DIM * index;

	const __global real* stateL = stateBuffer + NUM_STATES * indexPrev;
	const __global real* stateR = stateBuffer + NUM_STATES * index;
	
	__global real* eigenvalues = eigenvaluesBuffer + NUM_STATES * interfaceIndex;
	__global real* eigenvectors = eigenvectorsBuffer + NUM_STATES * NUM_STATES * interfaceIndex;
	__global real* eigenvectorsInverse = eigenvectorsInverseBuffer + NUM_STATES * NUM_STATES * interfaceIndex;

	real densityL = stateL[STATE_DENSITY];
	real invDensityL = 1.f / densityL;
	real4 velocityL = VELOCITY(stateL);
	real energyTotalL = stateL[STATE_ENERGY_TOTAL] * invDensityL;
	real energyKineticL = .5f * dot(velocityL, velocityL);
	real energyPotentialL = gravityPotentialBuffer[indexPrev];
	real energyInternalL = energyTotalL - energyKineticL - energyPotentialL;
	real pressureL = (GAMMA - 1.f) * densityL * energyInternalL;
	real enthalpyTotalL = energyTotalL + pressureL * invDensityL;
	real roeWeightL = sqrt(densityL);
	real4 magneticFieldL = (real4)(stateL.s5, stateL.s6, stateL.s7, 0.f);

	real densityR = stateR[STATE_DENSITY];
	real invDensityR = 1.f / densityR;
	real4 velocityR = (real4)(stateR.s1, stateR.s2, stateR.s3, 0.f) * invDensityR;
	real energyTotalR = stateR[STATE_ENERGY_TOTAL] * invDensityR;
	real energyKineticR = .5f * dot(velocityR, velocityR);
	real energyPotentialR = gravityPotentialBuffer[index];
	real energyInternalR = energyTotalR - energyKineticR - energyPotentialR;
	real pressureR = (GAMMA - 1.f) * densityR * energyInternalR;
	real enthalpyTotalR = energyTotalR + pressureR * invDensityR;
	real roeWeightR = sqrt(densityR);
	real4 magneticFieldR = (real4)(stateR.s5, stateR.s6, stateR.s7, 0.f);

	real roeWeightNormalization = 1.f / (roeWeightL + roeWeightR);
	real density = roeWeightL * roeWeightR;
	real4 velocity = (roeWeightL * velocityL + roeWeightR * velocityR) * roeWeightNormalization;
	real enthalpyTotal = (roeWeightL * enthalpyTotalL + roeWeightR * enthalpyTotalR) * roeWeightNormalization;
	real4 magneticField = (roeWeightL * magneticFieldL + roeWeightR * magneticFieldR) * roeWeightNormalization;
	
	real velocitySq = dot(velocity, velocity);
	real speedOfSound = sqrt((GAMMA - 1.f) * (enthalpyTotal - .5f * velocitySq));
	real speedOfSoundSq = speedOfSound * speedOfSound;
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
	eigenvalues[0] = velocityN - fastSpeed;
	eigenvalues[1] = velocityN - AlfvenSpeed;
	eigenvalues[2] = velocityN - slowSpeed;
	eigenvalues[3] = velocityN;
	eigenvalues[4] = velocityN;
	eigenvalues[5] = velocityN + slowSpeed;
	eigenvalues[6] = velocityN + AlfvenSpeed;
	eigenvalues[7] = velocityN + fastSpeed;
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
	//fast magnetoacoustic col 
	eigenvectorsA.s0 = (density * fastSpeedSq - magneticFieldSq) / speedOfSoundSq;
	eigenvectorsA.s8 = -fastSpeed * normal.x + magneticField.x * magneticFieldN / (density * fastSpeed);
	eigenvectorsB.s0 = -fastSpeed * normal.y + magneticField.y * magneticFieldN / (density * fastSpeed);
	eigenvectorsB.s8 = -fastSpeed * normal.z + magneticField.z * magneticFieldN / (density * fastSpeed);
	eigenvectorsC.s0 = magneticField.x - normal.x * magneticFieldN;
	eigenvectorsC.s8 = magneticField.y - normal.y * magneticFieldN;
	eigenvectorsD.s0 = magneticField.z - normal.z * magneticFieldN;
	eigenvectorsD.s8 = density * fastSpeedSq - magneticFieldSq;
	//Alfven col
	eigenvectorsA.s1 = 0.f;
	eigenvectorsA.s9 = magneticFieldCrossN.x * magneticFieldNSign;
	eigenvectorsB.s1 = magneticFieldCrossN.y * magneticFieldNSign;
	eigenvectorsB.s9 = magneticFieldCrossN.z * magneticFieldNSign;
	eigenvectorsC.s1 = magneticFieldCrossN.x * sqrtDensity;
	eigenvectorsC.s9 = magneticFieldCrossN.y * sqrtDensity;
	eigenvectorsD.s1 = magneticFieldCrossN.z * sqrtDensity;
	eigenvectorsD.s9 = 0.f;
	//slow magnetoacoustic col
	eigenvectorsA.s2 = (density * slowSpeedSq - magneticFieldSq) / speedOfSoundSq;
	eigenvectorsA.sA = -normal.x * slowSpeed + magneticField.x * magneticFieldN / (density * slowSpeed);
	eigenvectorsB.s2 = -normal.y * slowSpeed + magneticField.y * magneticFieldN / (density * slowSpeed);
	eigenvectorsB.sA = -normal.z * slowSpeed + magneticField.z * magneticFieldN / (density * slowSpeed);
	eigenvectorsC.s2 = magneticField.x - normal.x * magneticFieldN;
	eigenvectorsC.sA = magneticField.y - normal.y * magneticFieldN;
	eigenvectorsD.s2 = magneticField.z - normal.z * magneticFieldN;
	eigenvectorsD.sA = density * slowSpeedSq - magneticFieldSq;
	//entropy col
	eigenvectorsA.s3 = 1.f;
	eigenvectorsA.sB = 0.f; 
	eigenvectorsB.s3 = 0.f;
	eigenvectorsB.sB = 0.f;
	eigenvectorsC.s3 = 0.f;
	eigenvectorsC.sB = 0.f;
	eigenvectorsD.s3 = 0.f;
	eigenvectorsD.sB = 0.f;	
	//entropy col 
	eigenvectorsA.s4 = 0.f;
	eigenvectorsA.sC = 0.f;
	eigenvectorsB.s4 = 0.f;
	eigenvectorsB.sC = 0.f;
	eigenvectorsC.s4 = normal.x;
	eigenvectorsC.sC = normal.y;
	eigenvectorsD.s4 = normal.z;
	eigenvectorsD.sC = 0.f;
	//slow magnetoacoustic col
	eigenvectorsA.s5 = (density * slowSpeedSq - magneticFieldSq) / speedOfSoundSq;
	eigenvectorsA.sD = normal.x * slowSpeed - magneticField.x * magneticFieldN / (density * slowSpeed);
	eigenvectorsB.s5 = normal.y * slowSpeed - magneticField.y * magneticFieldN / (density * slowSpeed);
	eigenvectorsB.sD = normal.z * slowSpeed - magneticField.z * magneticFieldN / (density * slowSpeed);
	eigenvectorsC.s5 = magneticField.x - normal.x * magneticFieldN;
	eigenvectorsC.sD = magneticField.y - normal.y * magneticFieldN;
	eigenvectorsD.s5 = magneticField.z - normal.z * magneticFieldN;
	eigenvectorsD.sD = density * slowSpeedSq - magneticFieldSq;
	//Alfven col
	eigenvectorsA.s6 = 0.f;
	eigenvectorsA.sE = -magneticFieldCrossN.x * magneticFieldNSign;
	eigenvectorsB.s6 = -magneticFieldCrossN.y * magneticFieldNSign;
	eigenvectorsB.sE = -magneticFieldCrossN.z * magneticFieldNSign;
	eigenvectorsC.s6 = magneticFieldCrossN.x * sqrtDensity;
	eigenvectorsC.sE = magneticFieldCrossN.y * sqrtDensity;
	eigenvectorsD.s6 = magneticFieldCrossN.z * sqrtDensity;
	eigenvectorsD.sE = 0.f;
	//fast magnetoacoustic col
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
		
	eigenvectorsInverseBuffer[0 + 4 * interfaceIndex] = (real16)(
		//fast magnetoacoustic row
		//Alfven row
	);
	eigenvectorsInverseBuffer[1 + 4 * interfaceIndex] = (real16)(
		//slow magnetoacoustic row
		//entropy
	);
	eigenvectorsInverseBuffer[2 + 4 * interfaceIndex] = (real16)(
		//entropy
		//slow magnetoacoustic row
	);
	eigenvectorsInverseBuffer[3 + 4 * interfaceIndex] = (real16)(
		//Alfven row
		//fast magnetoacoustic row
	);
}

__kernel void calcEigenBasis(
	__global real* eigenvaluesBuffer,
	__global real* eigenvectorsBuffer,
	__global real* eigenvectorsInverseBuffer,
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
	calcEigenBasisSide(eigenvaluesBuffer, eigenvectorsBuffer, eigenvectorsInverseBuffer, stateBuffer, gravityPotentialBuffer, 0);
#if DIM > 1
	calcEigenBasisSide(eigenvaluesBuffer, eigenvectorsBuffer, eigenvectorsInverseBuffer, stateBuffer, gravityPotentialBuffer, 1);
#endif
#if DIM > 2
	calcEigenBasisSide(eigenvaluesBuffer, eigenvectorsBuffer, eigenvectorsInverseBuffer, stateBuffer, gravityPotentialBuffer, 2);
#endif
}

