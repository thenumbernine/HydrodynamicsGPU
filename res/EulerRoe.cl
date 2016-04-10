/*
The components of the Roe solver specific to the Euler equations
paritcularly the spectral decomposition
*/

#include "HydroGPU/Shared/Common.h"

#define gamma idealGas_heatCapacityRatio	//laziness

void calcEigenBasisSide(
	__global real* eigenvaluesBuffer,
	__global real* eigenvectorsBuffer,
	const __global real* stateBuffer,
	const __global real* potentialBuffer,
	const __global char* solidBuffer,
	int side);

void calcEigenBasisSide(
	__global real* eigenvaluesBuffer,
	__global real* eigenvectorsBuffer,
	const __global real* stateBuffer,
	const __global real* potentialBuffer,
	const __global char* solidBuffer,
	int side)
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
	int indexPrev = index - stepsize[side];

	const __global real* stateL = stateBuffer + NUM_STATES * indexPrev;
	const __global real* stateR = stateBuffer + NUM_STATES * index;
	
	int interfaceIndex = side + DIM * index;
	
	__global real* eigenvalues = eigenvaluesBuffer + NUM_STATES * interfaceIndex;
	__global real* eigenvectorsInverse = eigenvectorsBuffer + EIGEN_TRANSFORM_STRUCT_SIZE * interfaceIndex;
	__global real* eigenvectors = eigenvectorsInverse + NUM_STATES * NUM_STATES;

	char solidL = solidBuffer[indexPrev];
	char solidR = solidBuffer[index];

	real densityL = stateL[STATE_DENSITY];
	real4 velocityL = VELOCITY(stateL);
	real densityR = stateR[STATE_DENSITY];
	real4 velocityR = VELOCITY(stateR);
	
	if (solidL && !solidR) {
		densityL = densityR;
		velocityL = velocityR;
		velocityL[side+1] = -velocityR[side+1];
	}
	if (solidR && !solidL) {
		densityR = densityL;
		velocityR = velocityL;
		velocityR[side+1] = -velocityL[side+1];
	}

	real invDensityL = 1.f / densityL;
	real energyTotalL = stateL[STATE_ENERGY_TOTAL] * invDensityL;
	real energyKineticL = .5f * dot(velocityL, velocityL);
	real energyPotentialL = potentialBuffer[indexPrev];
	real energyInternalL = energyTotalL - energyKineticL - energyPotentialL;
	real pressureL = (gamma - 1.f) * densityL * energyInternalL;
	real enthalpyTotalL = energyTotalL + pressureL * invDensityL;		//<- should I remove potential energy from hTotal?
	real roeWeightL = sqrt(densityL);

	real invDensityR = 1.f / densityR;
	real energyTotalR = stateR[STATE_ENERGY_TOTAL] * invDensityR;
	real energyKineticR = .5f * dot(velocityR, velocityR);
	real energyPotentialR = potentialBuffer[index];
	real energyInternalR = energyTotalR - energyKineticR - energyPotentialR;
	real pressureR = (gamma - 1.f) * densityR * energyInternalR;
	real enthalpyTotalR = energyTotalR + pressureR * invDensityR;
	real roeWeightR = sqrt(densityR);

	real roeWeightNormalization = 1.f / (roeWeightL + roeWeightR);
	
	//variables used for eigenvalue and eigenvector calculations:
	real4 velocity = (roeWeightL * velocityL + roeWeightR * velocityR) * roeWeightNormalization;
	real enthalpyTotal = (roeWeightL * enthalpyTotalL + roeWeightR * enthalpyTotalR) * roeWeightNormalization;
	//...not these two:
	real energyPotential = (roeWeightL * energyPotentialL + roeWeightR * energyPotentialR) * roeWeightNormalization; 
	real velocitySq = dot(velocity, velocity);
	//...but this one:	
	real speedOfSound = sqrt((enthalpyTotal - .5f * velocitySq - energyPotential) * (gamma - 1.f));


//calculate flux in x-axis and rotate into normal
//works a bit more accurately than above
#if 1

#if DIM > 1
	if (side == 1) {
		velocity.xy = velocity.yx;
	} 
#if DIM > 2
	else if (side == 2) {
		velocity.xz = velocity.zx;
	}
#endif
#endif

	//eigenvalues

	eigenvalues[0] = velocity.x - speedOfSound;
	eigenvalues[1] = velocity.x;
#if DIM > 1
	eigenvalues[2] = velocity.x;
#if DIM > 2
	eigenvalues[3] = velocity.x;
#endif
#endif
	eigenvalues[DIM+1] = velocity.x + speedOfSound;

	//eigenvectors

	//min col 
	eigenvectors[0 + NUM_STATES * 0] = 1.f;
	eigenvectors[1 + NUM_STATES * 0] = velocity.x - speedOfSound;
#if DIM > 1
	eigenvectors[2 + NUM_STATES * 0] = velocity.y;
#if DIM > 2
	eigenvectors[3 + NUM_STATES * 0] = velocity.z;
#endif
#endif
	eigenvectors[(DIM+1) + NUM_STATES * 0] = enthalpyTotal - speedOfSound * velocity.x;
	//mid col (normal)
	eigenvectors[0 + NUM_STATES * 1] = 1.f;
	eigenvectors[1 + NUM_STATES * 1] = velocity.x;
#if DIM > 1
	eigenvectors[2 + NUM_STATES * 1] = velocity.y;
#if DIM > 2
	eigenvectors[3 + NUM_STATES * 1] = velocity.z;
#endif
#endif
	eigenvectors[(DIM+1) + NUM_STATES * 1] = .5f * velocitySq;
	//mid col (tangent A)
#if DIM > 1
	eigenvectors[0 + NUM_STATES * 2] = 0.f;
	eigenvectors[1 + NUM_STATES * 2] = 0.f;
	eigenvectors[2 + NUM_STATES * 2] = 1.f;
#if DIM > 2
	eigenvectors[3 + NUM_STATES * 2] = 0.f;
#endif
	eigenvectors[(DIM+1) + NUM_STATES * 2] = velocity.y;
#endif
	//mid col (tangent B)
#if DIM > 2
	eigenvectors[0 + NUM_STATES * 3] = 0.f;
	eigenvectors[1 + NUM_STATES * 3] = 0.f;
	eigenvectors[2 + NUM_STATES * 3] = 0.f;
	eigenvectors[3 + NUM_STATES * 3] = 1.f;
	eigenvectors[(DIM+1) + NUM_STATES * 3] = velocity.z;
#endif
	//max col 
	eigenvectors[0 + NUM_STATES * (DIM+1)] = 1.f;
	eigenvectors[1 + NUM_STATES * (DIM+1)] = velocity.x + speedOfSound;
#if DIM > 1
	eigenvectors[2 + NUM_STATES * (DIM+1)] = velocity.y;
#if DIM > 2
	eigenvectors[3 + NUM_STATES * (DIM+1)] = velocity.z;
#endif
#endif
	eigenvectors[(DIM+1) + NUM_STATES * (DIM+1)] = enthalpyTotal + speedOfSound * velocity.x;

	
	//calculate eigenvector inverses ... 
	real invDenom = .5f / (speedOfSound * speedOfSound);
	
	//min row
	eigenvectorsInverse[0 + NUM_STATES * 0] = (.5f * (gamma - 1.f) * velocitySq + speedOfSound * velocity.x) * invDenom;
	eigenvectorsInverse[0 + NUM_STATES * 1] = -(speedOfSound + (gamma - 1.f) * velocity.x) * invDenom;
#if DIM > 1
	eigenvectorsInverse[0 + NUM_STATES * 2] = -(gamma - 1.f) * velocity.y * invDenom;
#if DIM > 2
	eigenvectorsInverse[0 + NUM_STATES * 3] = -(gamma - 1.f) * velocity.z * invDenom;
#endif
#endif
	eigenvectorsInverse[0 + NUM_STATES * (DIM+1)] = (gamma - 1.f) * invDenom;
	//mid normal row
	eigenvectorsInverse[1 + NUM_STATES * 0] = 1.f - (gamma - 1.f) * velocitySq * invDenom;
	eigenvectorsInverse[1 + NUM_STATES * 1] = (gamma - 1.f) * velocity.x * 2.f * invDenom;
#if DIM > 1
	eigenvectorsInverse[1 + NUM_STATES * 2] = (gamma - 1.f) * velocity.y * 2.f * invDenom;
#if DIM > 2
	eigenvectorsInverse[1 + NUM_STATES * 3] = (gamma - 1.f) * velocity.z * 2.f * invDenom;
#endif
#endif
	eigenvectorsInverse[1 + NUM_STATES * (DIM+1)] = -(gamma - 1.f) * 2.f * invDenom;
	//mid tangent A row
#if DIM > 1
	eigenvectorsInverse[2 + NUM_STATES * 0] = -velocity.y; 
	eigenvectorsInverse[2 + NUM_STATES * 1] = 0.f;
	eigenvectorsInverse[2 + NUM_STATES * 2] = 1.f;
#if DIM > 2
	eigenvectorsInverse[2 + NUM_STATES * 3] = 0.f;
#endif
	eigenvectorsInverse[2 + NUM_STATES * (DIM+1)] = 0.f;
#endif
	//mid tangent B row
#if DIM > 2
	eigenvectorsInverse[3 + NUM_STATES * 0] = -velocity.z;
	eigenvectorsInverse[3 + NUM_STATES * 1] = 0.f;
	eigenvectorsInverse[3 + NUM_STATES * 2] = 0.f;
	eigenvectorsInverse[3 + NUM_STATES * 3] = 1.f;
	eigenvectorsInverse[3 + NUM_STATES * (DIM+1)] = 0.f;
#endif
	//max row
	eigenvectorsInverse[(DIM+1) + NUM_STATES * 0] = (.5f * (gamma - 1.f) * velocitySq - speedOfSound * velocity.x) * invDenom;
	eigenvectorsInverse[(DIM+1) + NUM_STATES * 1] = (speedOfSound - (gamma - 1.f) * velocity.x) * invDenom;
#if DIM > 1
	eigenvectorsInverse[(DIM+1) + NUM_STATES * 2] = -(gamma - 1.f) * velocity.y * invDenom;
#if DIM > 2
	eigenvectorsInverse[(DIM+1) + NUM_STATES * 3] = -(gamma - 1.f) * velocity.z * invDenom;
#endif
#endif
	eigenvectorsInverse[(DIM+1) + NUM_STATES * (DIM+1)] = (gamma - 1.f) * invDenom;

#if DIM > 1
	if (side == 1) {
		for (int i = 0; i < NUM_STATES; ++i) {
			real tmp;
			//each row's xy <- yx
			tmp = eigenvectorsInverse[i + NUM_STATES * STATE_MOMENTUM_X];
			eigenvectorsInverse[i + NUM_STATES * STATE_MOMENTUM_X] = eigenvectorsInverse[i + NUM_STATES * STATE_MOMENTUM_Y];
			eigenvectorsInverse[i + NUM_STATES * STATE_MOMENTUM_Y] = tmp;
			//each column's xy <- yx
			tmp = eigenvectors[STATE_MOMENTUM_X + NUM_STATES * i];
			eigenvectors[STATE_MOMENTUM_X + NUM_STATES * i] = eigenvectors[STATE_MOMENTUM_Y + NUM_STATES * i];
			eigenvectors[STATE_MOMENTUM_Y + NUM_STATES * i] = tmp;
		}
	}
#if DIM > 2
	else if (side == 2) {
		for (int i = 0; i < NUM_STATES; ++i) {
			real tmp;
			tmp = eigenvectorsInverse[i + NUM_STATES * STATE_MOMENTUM_X];
			eigenvectorsInverse[i + NUM_STATES * STATE_MOMENTUM_X] = eigenvectorsInverse[i + NUM_STATES * STATE_MOMENTUM_Z];
			eigenvectorsInverse[i + NUM_STATES * STATE_MOMENTUM_Z] = tmp;
			tmp = eigenvectors[STATE_MOMENTUM_X + NUM_STATES * i];
			eigenvectors[STATE_MOMENTUM_X + NUM_STATES * i] = eigenvectors[STATE_MOMENTUM_Z + NUM_STATES * i];
			eigenvectors[STATE_MOMENTUM_Z + NUM_STATES * i] = tmp;
		}
	}
#endif
#endif
	
#endif

}

__kernel void calcEigenBasis(
	__global real* eigenvaluesBuffer,
	__global real* eigenvectorsBuffer,
	const __global real* stateBuffer,
	const __global real* potentialBuffer,
	const __global char* solidBuffer)
{
	for (int side = 0; side < DIM; ++side) {
		calcEigenBasisSide(eigenvaluesBuffer, eigenvectorsBuffer, stateBuffer, potentialBuffer, solidBuffer, side);
	}
}
