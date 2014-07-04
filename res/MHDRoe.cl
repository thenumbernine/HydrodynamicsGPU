/*
The components of the Roe solver specific to the MHD equations
paritcularly the spectral decomposition
*/

#include "HydroGPU/Shared/Common.h"

void mat88invert(
	__global real* matrixInverse,
	const __global real* matrix);

void mat88invert(
	__global real* matrixInverse,
	const __global real* matrix)
{
	int indxc[NUM_STATES], indxr[NUM_STATES], ipiv[NUM_STATES];
	int i, icol = 0, irow = 0, j, k, l, ll;
	real big, dum, pivinv, temp;


	for (int i = 0; i < NUM_STATES * NUM_STATES; ++i) {
		matrixInverse[i] = matrix[i];
	}

	for (j = 0; j < NUM_STATES; ++j) {
		ipiv[j] = 0;
	}
	
	for (i = 0; i < NUM_STATES; ++i) {
		big = 0.f;
		for ( j = 0; j < NUM_STATES; ++j) {
			if (ipiv[j] != 1) {
				for (k = 0; k < NUM_STATES; ++k) {
					if (ipiv[k] == 0) {
						if (fabs(matrixInverse[j + NUM_STATES * k]) >= big) {
							big = fabs(matrixInverse[j + NUM_STATES * k]);
							irow = j;
							icol = k;
						}
					} else {
						if(ipiv[k] > 1) {
							return;	//throw Common::Exception() << "singular!";
						}
					}
				}
			}
		}
		++ipiv[icol];
		if(irow != icol) {
			for (int k = 0; k < NUM_STATES; ++k) {
				temp = matrixInverse[irow + NUM_STATES * k];
				matrixInverse[irow + NUM_STATES * k] = matrixInverse[icol + NUM_STATES * k];
				matrixInverse[icol + NUM_STATES * k] = temp;
			}
		}
		indxr[i] = irow;
		indxc[i] = icol;
		if(matrixInverse[icol + NUM_STATES * icol] == 0.f) {
			return;	//throw Common::Exception() << "singular!";
		}
		pivinv = 1.f / matrixInverse[icol + NUM_STATES * icol];
		matrixInverse[icol + NUM_STATES * icol] = 1.f;
		for (l = 0; l < NUM_STATES; ++l) {
			matrixInverse[icol + NUM_STATES * l] *= pivinv;
		}
		for (ll = 0; ll < NUM_STATES; ++ll) {
			if (ll != icol) {
				dum = matrixInverse[ll + NUM_STATES * icol];
				matrixInverse[ll + NUM_STATES * icol] = 0.f;
				for (l = 0; l < NUM_STATES; ++l) {
					matrixInverse[ll + NUM_STATES * l] -= matrixInverse[icol + NUM_STATES * l] * dum;
				}
			}
		}
	}
	
	for (l = NUM_STATES-1; l >= 0; --l) {
		if (indxr[l] != indxc[l]) {
			for (k = 0; k < NUM_STATES; ++k) {
				temp = matrixInverse[k + NUM_STATES * indxr[l]];
				matrixInverse[k + NUM_STATES * indxr[l]] = matrixInverse[k + NUM_STATES * indxc[l]];
				matrixInverse[k + NUM_STATES * indxc[l]] = temp;
			}
		}
	}
}

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
	int indexPrev = index - stepsize[side];
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
	real4 magneticFieldL = (real4)(stateL[STATE_MAGNETIC_FIELD_X], stateL[STATE_MAGNETIC_FIELD_Y], stateL[STATE_MAGNETIC_FIELD_Z], 0.f);

	real densityR = stateR[STATE_DENSITY];
	real invDensityR = 1.f / densityR;
	real4 velocityR = VELOCITY(stateR);
	real energyTotalR = stateR[STATE_ENERGY_TOTAL] * invDensityR;
	real energyKineticR = .5f * dot(velocityR, velocityR);
	real energyPotentialR = gravityPotentialBuffer[index];
	real energyInternalR = energyTotalR - energyKineticR - energyPotentialR;
	real pressureR = (GAMMA - 1.f) * densityR * energyInternalR;
	real enthalpyTotalR = energyTotalR + pressureR * invDensityR;
	real roeWeightR = sqrt(densityR);
	real4 magneticFieldR = (real4)(stateR[STATE_MAGNETIC_FIELD_X], stateR[STATE_MAGNETIC_FIELD_Y], stateR[STATE_MAGNETIC_FIELD_Z], 0.f);

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
	real discrA = max(0.f, starSpeedToTheFourth - speedOfSoundSq * AlfvenSpeedSq);
	real discrB = sqrt(discrA);
	real slowSpeedSq = max(0.f, starSpeedSq - discrB);
	real slowSpeed = sqrt(slowSpeedSq);
	real fastSpeedSq = max(0.f, starSpeedSq + discrB);
	real fastSpeed = sqrt(fastSpeedSq);

	real4 tangentA = (real4)(normal.y, normal.z, normal.x, 0.f);
	real4 tangentB = (real4)(normal.z, normal.x, normal.y, 0.f);
	real velocityN = dot(velocity, normal);
	real velocityTA = dot(velocity, tangentA);
	real velocityTB = dot(velocity, tangentB);

	//eigenvalues

	eigenvalues[0] = velocityN - fastSpeed;
	eigenvalues[1] = velocityN - AlfvenSpeed;
	eigenvalues[2] = velocityN - slowSpeed;
	eigenvalues[3] = velocityN;
	eigenvalues[4] = velocityN;
	eigenvalues[5] = velocityN + slowSpeed;
	eigenvalues[6] = velocityN + AlfvenSpeed;
	eigenvalues[7] = velocityN + fastSpeed;

	//eigenvectors

	real sqrtDensity = sqrt(density);
	real magneticFieldNSign = step(0.f, magneticFieldN) * 2.f - 1.f;
	real4 magneticFieldCrossN = cross(magneticField, normal);
	//fast magnetoacoustic col 
	eigenvectors[0 + NUM_STATES * 0] = (density * fastSpeedSq - magneticFieldSq) / speedOfSoundSq;
	eigenvectors[1 + NUM_STATES * 0] = -fastSpeed * normal.x + magneticField.x * magneticFieldN / (density * fastSpeed);
	eigenvectors[2 + NUM_STATES * 0] = -fastSpeed * normal.y + magneticField.y * magneticFieldN / (density * fastSpeed);
	eigenvectors[3 + NUM_STATES * 0] = -fastSpeed * normal.z + magneticField.z * magneticFieldN / (density * fastSpeed);
	eigenvectors[4 + NUM_STATES * 0] = magneticField.x - normal.x * magneticFieldN;
	eigenvectors[5 + NUM_STATES * 0] = magneticField.y - normal.y * magneticFieldN;
	eigenvectors[6 + NUM_STATES * 0] = magneticField.z - normal.z * magneticFieldN;
	eigenvectors[7 + NUM_STATES * 0] = density * fastSpeedSq - magneticFieldSq;
	//Alfven col
	eigenvectors[0 + NUM_STATES * 1] = 0.f;
	eigenvectors[1 + NUM_STATES * 1] = magneticFieldCrossN.x * magneticFieldNSign;
	eigenvectors[2 + NUM_STATES * 1] = magneticFieldCrossN.y * magneticFieldNSign;
	eigenvectors[3 + NUM_STATES * 1] = magneticFieldCrossN.z * magneticFieldNSign;
	eigenvectors[4 + NUM_STATES * 1] = magneticFieldCrossN.x * sqrtDensity;
	eigenvectors[5 + NUM_STATES * 1] = magneticFieldCrossN.y * sqrtDensity;
	eigenvectors[6 + NUM_STATES * 1] = magneticFieldCrossN.z * sqrtDensity;
	eigenvectors[7 + NUM_STATES * 1] = 0.f;
	//slow magnetoacoustic col
	eigenvectors[0 + NUM_STATES * 2] = (density * slowSpeedSq - magneticFieldSq) / speedOfSoundSq;
	eigenvectors[1 + NUM_STATES * 2] = -normal.x * slowSpeed + magneticField.x * magneticFieldN / (density * slowSpeed);
	eigenvectors[2 + NUM_STATES * 2] = -normal.y * slowSpeed + magneticField.y * magneticFieldN / (density * slowSpeed);
	eigenvectors[3 + NUM_STATES * 2] = -normal.z * slowSpeed + magneticField.z * magneticFieldN / (density * slowSpeed);
	eigenvectors[4 + NUM_STATES * 2] = magneticField.x - normal.x * magneticFieldN;
	eigenvectors[5 + NUM_STATES * 2] = magneticField.y - normal.y * magneticFieldN;
	eigenvectors[6 + NUM_STATES * 2] = magneticField.z - normal.z * magneticFieldN;
	eigenvectors[7 + NUM_STATES * 2] = density * slowSpeedSq - magneticFieldSq;
	//entropy col
	eigenvectors[0 + NUM_STATES * 3] = 1.f;
	eigenvectors[1 + NUM_STATES * 3] = 0.f; 
	eigenvectors[2 + NUM_STATES * 3] = 0.f;
	eigenvectors[3 + NUM_STATES * 3] = 0.f;
	eigenvectors[4 + NUM_STATES * 3] = 0.f;
	eigenvectors[5 + NUM_STATES * 3] = 0.f;
	eigenvectors[6 + NUM_STATES * 3] = 0.f;
	eigenvectors[7 + NUM_STATES * 3] = 0.f;	
	//entropy col 
	eigenvectors[0 + NUM_STATES * 4] = 0.f;
	eigenvectors[1 + NUM_STATES * 4] = 0.f;
	eigenvectors[2 + NUM_STATES * 4] = 0.f;
	eigenvectors[3 + NUM_STATES * 4] = 0.f;
	eigenvectors[4 + NUM_STATES * 4] = normal.x;
	eigenvectors[5 + NUM_STATES * 4] = normal.y;
	eigenvectors[6 + NUM_STATES * 4] = normal.z;
	eigenvectors[7 + NUM_STATES * 4] = 0.f;
	//slow magnetoacoustic col
	eigenvectors[0 + NUM_STATES * 5] = (density * slowSpeedSq - magneticFieldSq) / speedOfSoundSq;
	eigenvectors[1 + NUM_STATES * 5] = normal.x * slowSpeed - magneticField.x * magneticFieldN / (density * slowSpeed);
	eigenvectors[2 + NUM_STATES * 5] = normal.y * slowSpeed - magneticField.y * magneticFieldN / (density * slowSpeed);
	eigenvectors[3 + NUM_STATES * 5] = normal.z * slowSpeed - magneticField.z * magneticFieldN / (density * slowSpeed);
	eigenvectors[4 + NUM_STATES * 5] = magneticField.x - normal.x * magneticFieldN;
	eigenvectors[5 + NUM_STATES * 5] = magneticField.y - normal.y * magneticFieldN;
	eigenvectors[6 + NUM_STATES * 5] = magneticField.z - normal.z * magneticFieldN;
	eigenvectors[7 + NUM_STATES * 5] = density * slowSpeedSq - magneticFieldSq;
	//Alfven col
	eigenvectors[0 + NUM_STATES * 6] = 0.f;
	eigenvectors[1 + NUM_STATES * 6] = -magneticFieldCrossN.x * magneticFieldNSign;
	eigenvectors[2 + NUM_STATES * 6] = -magneticFieldCrossN.y * magneticFieldNSign;
	eigenvectors[3 + NUM_STATES * 6] = -magneticFieldCrossN.z * magneticFieldNSign;
	eigenvectors[4 + NUM_STATES * 6] = magneticFieldCrossN.x * sqrtDensity;
	eigenvectors[5 + NUM_STATES * 6] = magneticFieldCrossN.y * sqrtDensity;
	eigenvectors[6 + NUM_STATES * 6] = magneticFieldCrossN.z * sqrtDensity;
	eigenvectors[7 + NUM_STATES * 6] = 0.f;
	//fast magnetoacoustic col
	eigenvectors[0 + NUM_STATES * 7] = (density * fastSpeedSq - magneticFieldSq) / speedOfSoundSq;
	eigenvectors[1 + NUM_STATES * 7] = normal.x * fastSpeed - magneticField.x * magneticFieldN / (density * fastSpeed);
	eigenvectors[2 + NUM_STATES * 7] = normal.y * fastSpeed - magneticField.y * magneticFieldN / (density * fastSpeed);
	eigenvectors[3 + NUM_STATES * 7] = normal.z * fastSpeed - magneticField.z * magneticFieldN / (density * fastSpeed);
	eigenvectors[4 + NUM_STATES * 7] = magneticField.x - normal.x * magneticFieldN;
	eigenvectors[5 + NUM_STATES * 7] = magneticField.y - normal.y * magneticFieldN;
	eigenvectors[6 + NUM_STATES * 7] = magneticField.z - normal.z * magneticFieldN;
	eigenvectors[7 + NUM_STATES * 7] = density * fastSpeedSq - magneticFieldSq;
	
	//eigenvectors inverse
	//either this or the eigenvectors themselves is not working
	mat88invert(eigenvectorsInverse, eigenvectors);
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

