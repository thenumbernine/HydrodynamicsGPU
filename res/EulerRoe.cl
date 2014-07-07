/*
The components of the Roe solver specific to the Euler equations
paritcularly the spectral decomposition
*/

#include "HydroGPU/Shared/Common.h"

void calcEigenBasisSide(
	__global real* eigenvaluesBuffer,
	__global real* eigenvectorsBuffer,
	__global real* eigenvectorsInverseBuffer,
	const __global real* stateBuffer,
	const __global real* potentialBuffer,
	int side);

void calcEigenBasisSide(
	__global real* eigenvaluesBuffer,
	__global real* eigenvectorsBuffer,
	__global real* eigenvectorsInverseBuffer,
	const __global real* stateBuffer,
	const __global real* potentialBuffer,
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
	real energyPotentialL = potentialBuffer[indexPrev];
	real energyInternalL = energyTotalL - energyKineticL - energyPotentialL;
	real pressureL = (GAMMA - 1.f) * densityL * energyInternalL;
	real enthalpyTotalL = energyTotalL + pressureL * invDensityL;
	real roeWeightL = sqrt(densityL);

	real densityR = stateR[STATE_DENSITY];
	real invDensityR = 1.f / densityR;
	real4 velocityR = VELOCITY(stateR);
	real energyTotalR = stateR[STATE_ENERGY_TOTAL] * invDensityR;
	real energyKineticR = .5f * dot(velocityR, velocityR);
	real energyPotentialR = potentialBuffer[index];
	real energyInternalR = energyTotalR - energyKineticR - energyPotentialR;
	real pressureR = (GAMMA - 1.f) * densityR * energyInternalR;
	real enthalpyTotalR = energyTotalR + pressureR * invDensityR;
	real roeWeightR = sqrt(densityR);

	real roeWeightNormalization = 1.f / (roeWeightL + roeWeightR);
	real4 velocity = (roeWeightL * velocityL + roeWeightR * velocityR) * roeWeightNormalization;
	real enthalpyTotal = (roeWeightL * enthalpyTotalL + roeWeightR * enthalpyTotalR) * roeWeightNormalization;
	real energyPotential = (roeWeightL * energyPotentialL + roeWeightR * energyPotentialR) * roeWeightNormalization; 
	
	real velocitySq = dot(velocity, velocity);
	real speedOfSound = sqrt((enthalpyTotal - .5f * velocitySq - energyPotential) * (GAMMA - 1.f));

//calculate flux based on normal
//contains subtle numerical errors along the x axis
#if 0	

	real4 normal = (real4)(0.f, 0.f, 0.f, 0.f);
	normal[side] = 1;

	real velocityN = dot(velocity, normal);
#if DIM > 1
	real4 tangentA = (real4)(normal.y, normal.z, normal.x, 0.f);
	real velocityTA = dot(velocity, tangentA);
#if DIM > 2
	real4 tangentB = (real4)(normal.z, normal.x, normal.y, 0.f);
	real velocityTB = dot(velocity, tangentB);
#endif
#endif

	//eigenvalues

	eigenvalues[0] = velocityN - speedOfSound;
	eigenvalues[1] = velocityN;
#if DIM > 1
	eigenvalues[2] = velocityN;
#if DIM > 2
	eigenvalues[3] = velocityN;
#endif
#endif
	eigenvalues[DIM+1] = velocityN + speedOfSound;

	//eigenvectors


	//min col 
	eigenvectors[0 + NUM_STATES * 0] = 1.f;
	eigenvectors[1 + NUM_STATES * 0] = velocity.x - speedOfSound * normal.x;
#if DIM > 1
	eigenvectors[2 + NUM_STATES * 0] = velocity.y - speedOfSound * normal.y;
#if DIM > 2
	eigenvectors[3 + NUM_STATES * 0] = velocity.z - speedOfSound * normal.z;
#endif
#endif
	eigenvectors[(DIM+1) + NUM_STATES * 0] = enthalpyTotal - speedOfSound * velocityN;
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
	eigenvectors[1 + NUM_STATES * 2] = tangentA.x;
	eigenvectors[2 + NUM_STATES * 2] = tangentA.y;
#if DIM > 2
	eigenvectors[3 + NUM_STATES * 2] = tangentA.z;
#endif
	eigenvectors[(DIM+1) + NUM_STATES * 2] = velocityTA;
#endif
	//mid col (tangent B)
#if DIM > 2
	eigenvectors[0 + NUM_STATES * 3] = 0.f;
	eigenvectors[1 + NUM_STATES * 3] = tangentB.x;
	eigenvectors[2 + NUM_STATES * 3] = tangentB.y;
	eigenvectors[3 + NUM_STATES * 3] = tangentB.z;
	eigenvectors[(DIM+1) + NUM_STATES * 3] = velocityTB;
#endif
	//max col 
	eigenvectors[0 + NUM_STATES * (DIM+1)] = 1.f;
	eigenvectors[1 + NUM_STATES * (DIM+1)] = velocity.x + speedOfSound * normal.x;
#if DIM > 1
	eigenvectors[2 + NUM_STATES * (DIM+1)] = velocity.y + speedOfSound * normal.y;
#if DIM > 2
	eigenvectors[3 + NUM_STATES * (DIM+1)] = velocity.z + speedOfSound * normal.z;
#endif
#endif
	eigenvectors[(DIM+1) + NUM_STATES * (DIM+1)] = enthalpyTotal + speedOfSound * velocityN;
	

	//calculate eigenvector inverses ... 
	real invDenom = .5f / (speedOfSound * speedOfSound);
	
	//min row
	eigenvectorsInverse[0 + NUM_STATES * 0] = (.5f * (GAMMA - 1.f) * velocitySq + speedOfSound * velocityN) * invDenom;
	eigenvectorsInverse[0 + NUM_STATES * 1] = -(normal.x * speedOfSound + (GAMMA - 1.f) * velocity.x) * invDenom;
#if DIM > 1
	eigenvectorsInverse[0 + NUM_STATES * 2] = -(normal.y * speedOfSound + (GAMMA - 1.f) * velocity.y) * invDenom;
#if DIM > 2
	eigenvectorsInverse[0 + NUM_STATES * 3] = -(normal.z * speedOfSound + (GAMMA - 1.f) * velocity.z) * invDenom;
#endif
#endif
	eigenvectorsInverse[0 + NUM_STATES * (DIM+1)] = (GAMMA - 1.f) * invDenom;
	//mid normal row
	eigenvectorsInverse[1 + NUM_STATES * 0] = 1.f - (GAMMA - 1.f) * velocitySq * invDenom;
	eigenvectorsInverse[1 + NUM_STATES * 1] = (GAMMA - 1.f) * velocity.x * 2.f * invDenom;
#if DIM > 1
	eigenvectorsInverse[1 + NUM_STATES * 2] = (GAMMA - 1.f) * velocity.y * 2.f * invDenom;
#if DIM > 2
	eigenvectorsInverse[1 + NUM_STATES * 3] = (GAMMA - 1.f) * velocity.z * 2.f * invDenom;
#endif
#endif
	eigenvectorsInverse[1 + NUM_STATES * (DIM+1)] = -(GAMMA - 1.f) * 2.f * invDenom;
	//mid tangent A row
#if DIM > 1
	eigenvectorsInverse[2 + NUM_STATES * 0] = -velocityTA; 
	eigenvectorsInverse[2 + NUM_STATES * 1] = tangentA.x;
	eigenvectorsInverse[2 + NUM_STATES * 2] = tangentA.y;
#if DIM > 2
	eigenvectorsInverse[2 + NUM_STATES * 3] = tangentA.z;
#endif
	eigenvectorsInverse[2 + NUM_STATES * (DIM+1)] = 0.f;
#endif
	//mid tangent B row
#if DIM > 2
	eigenvectorsInverse[3 + NUM_STATES * 0] = -velocityTB;
	eigenvectorsInverse[3 + NUM_STATES * 1] = tangentB.x;
	eigenvectorsInverse[3 + NUM_STATES * 2] = tangentB.y;
	eigenvectorsInverse[3 + NUM_STATES * 3] = tangentB.z;
	eigenvectorsInverse[3 + NUM_STATES * (DIM+1)] = 0.f;
#endif
	//max row
	eigenvectorsInverse[(DIM+1) + NUM_STATES * 0] = (.5f * (GAMMA - 1.f) * velocitySq - speedOfSound * velocityN) * invDenom;
	eigenvectorsInverse[(DIM+1) + NUM_STATES * 1] = (normal.x * speedOfSound - (GAMMA - 1.f) * velocity.x) * invDenom;
#if DIM > 1
	eigenvectorsInverse[(DIM+1) + NUM_STATES * 2] = (normal.y * speedOfSound - (GAMMA - 1.f) * velocity.y) * invDenom;
#if DIM > 2
	eigenvectorsInverse[(DIM+1) + NUM_STATES * 3] = (normal.z * speedOfSound - (GAMMA - 1.f) * velocity.z) * invDenom;
#endif
#endif
	eigenvectorsInverse[(DIM+1) + NUM_STATES * (DIM+1)] = (GAMMA - 1.f) * invDenom;
#endif




//calculate flux in x-axis and rotate into normal
//works a bit more accurately than above
#if 1

#if DIM > 1
	if (side == 1) {
		velocity = (real4)(velocity.y, -velocity.x, velocity.z, 0.f);	// -90' rotation to put the y axis contents into the x axis
	} 
#if DIM > 2
	else if (side == 2) {
		velocity = (real4)(velocity.z, velocity.y, -velocity.x, 0.f);	//-90' rotation to put the z axis in the x axis
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
	eigenvectorsInverse[0 + NUM_STATES * 0] = (.5f * (GAMMA - 1.f) * velocitySq + speedOfSound * velocity.x) * invDenom;
	eigenvectorsInverse[0 + NUM_STATES * 1] = -(speedOfSound + (GAMMA - 1.f) * velocity.x) * invDenom;
#if DIM > 1
	eigenvectorsInverse[0 + NUM_STATES * 2] = -(GAMMA - 1.f) * velocity.y * invDenom;
#if DIM > 2
	eigenvectorsInverse[0 + NUM_STATES * 3] = -(GAMMA - 1.f) * velocity.z * invDenom;
#endif
#endif
	eigenvectorsInverse[0 + NUM_STATES * (DIM+1)] = (GAMMA - 1.f) * invDenom;
	//mid normal row
	eigenvectorsInverse[1 + NUM_STATES * 0] = 1.f - (GAMMA - 1.f) * velocitySq * invDenom;
	eigenvectorsInverse[1 + NUM_STATES * 1] = (GAMMA - 1.f) * velocity.x * 2.f * invDenom;
#if DIM > 1
	eigenvectorsInverse[1 + NUM_STATES * 2] = (GAMMA - 1.f) * velocity.y * 2.f * invDenom;
#if DIM > 2
	eigenvectorsInverse[1 + NUM_STATES * 3] = (GAMMA - 1.f) * velocity.z * 2.f * invDenom;
#endif
#endif
	eigenvectorsInverse[1 + NUM_STATES * (DIM+1)] = -(GAMMA - 1.f) * 2.f * invDenom;
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
	eigenvectorsInverse[(DIM+1) + NUM_STATES * 0] = (.5f * (GAMMA - 1.f) * velocitySq - speedOfSound * velocity.x) * invDenom;
	eigenvectorsInverse[(DIM+1) + NUM_STATES * 1] = (speedOfSound - (GAMMA - 1.f) * velocity.x) * invDenom;
#if DIM > 1
	eigenvectorsInverse[(DIM+1) + NUM_STATES * 2] = -(GAMMA - 1.f) * velocity.y * invDenom;
#if DIM > 2
	eigenvectorsInverse[(DIM+1) + NUM_STATES * 3] = -(GAMMA - 1.f) * velocity.z * invDenom;
#endif
#endif
	eigenvectorsInverse[(DIM+1) + NUM_STATES * (DIM+1)] = (GAMMA - 1.f) * invDenom;

#if DIM > 1
	if (side == 1) {
		for (int i = 0; i < NUM_STATES; ++i) {
			real tmp;

			//-90' rotation applied to the LHS of incoming velocity vectors, to move their y axis into the x axis
			// is equivalent of a -90' rotation applied to the RHS of the flux jacobian A
			// and A = Q V Q-1 for Q = the right eigenvectors and Q-1 the left eigenvectors
			// so a -90' rotation applied to the RHS of A is a +90' rotation applied to the RHS of Q-1 the left eigenvectors
			//and while a rotation applied to the LHS of a vector rotates the elements of its column vectors, a rotation applied to the RHS rotates the elements of its row vectors 
			//each row's y <- x, x <- -y
			tmp = eigenvectorsInverse[i + NUM_STATES * STATE_MOMENTUM_X];
			eigenvectorsInverse[i + NUM_STATES * STATE_MOMENTUM_X] = -eigenvectorsInverse[i + NUM_STATES * STATE_MOMENTUM_Y];
			eigenvectorsInverse[i + NUM_STATES * STATE_MOMENTUM_Y] = tmp;
			//a -90' rotation applied to the RHS of A must be corrected with a 90' rotation on the LHS of A
			//this rotates the elements of the column vectors by 90'
			//each column's x <- y, y <- -x
			tmp = eigenvectors[STATE_MOMENTUM_X + NUM_STATES * i];
			eigenvectors[STATE_MOMENTUM_X + NUM_STATES * i] = -eigenvectors[STATE_MOMENTUM_Y + NUM_STATES * i];
			eigenvectors[STATE_MOMENTUM_Y + NUM_STATES * i] = tmp;
		}
	}
#if DIM > 2
	else if (side == 2) {
		for (int i = 0; i < NUM_STATES; ++i) {
			real tmp;
			tmp = eigenvectorsInverse[i + NUM_STATES * STATE_MOMENTUM_X];
			eigenvectorsInverse[i + NUM_STATES * STATE_MOMENTUM_X] = -eigenvectorsInverse[i + NUM_STATES * STATE_MOMENTUM_Z];
			eigenvectorsInverse[i + NUM_STATES * STATE_MOMENTUM_Z] = tmp;
			tmp = eigenvectors[STATE_MOMENTUM_X + NUM_STATES * i];
			eigenvectors[STATE_MOMENTUM_X + NUM_STATES * i] = -eigenvectors[STATE_MOMENTUM_Z + NUM_STATES * i];
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
	__global real* eigenvectorsInverseBuffer,
	const __global real* stateBuffer,
	const __global real* potentialBuffer)
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
	calcEigenBasisSide(eigenvaluesBuffer, eigenvectorsBuffer, eigenvectorsInverseBuffer, stateBuffer, potentialBuffer, 0);
#if DIM > 1
	calcEigenBasisSide(eigenvaluesBuffer, eigenvectorsBuffer, eigenvectorsInverseBuffer, stateBuffer, potentialBuffer, 1);
#endif
#if DIM > 2
	calcEigenBasisSide(eigenvaluesBuffer, eigenvectorsBuffer, eigenvectorsInverseBuffer, stateBuffer, potentialBuffer, 2);
#endif
}


