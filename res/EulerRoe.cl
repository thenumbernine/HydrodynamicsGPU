/*
The components of the Roe solver specific to the Euler equations
paritcularly the spectral decomposition
*/

#include "HydroGPU/Shared/Common.h"

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

		real densityR = stateR.s0;
		real invDensityR = 1.f / densityR;
		real4 velocityR = (real4)(stateR.s1, stateR.s2, stateR.s3, 0.f) * invDensityR;
		real energyTotalR = stateR.s4 * invDensityR;

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
		real4 velocity = (weightL * velocityL + weightR * velocityR) * roeWeightNormalization;
		real enthalpyTotal = (weightL * enthalpyTotalL + weightR * enthalpyTotalR) * roeWeightNormalization;
		
		real velocitySq = dot(velocity, velocity);
		real speedOfSound = sqrt((GAMMA - 1.f) * (enthalpyTotal - .5f * velocitySq));
	
#if 1	//calculate flux based on normal

		real4 normal = (real4)(0.f, 0.f, 0.f, 0.f);
		normal[side] = 1;
	
		real4 tangentA = (real4)(normal.y, normal.z, normal.x, 0.f);
		real4 tangentB = (real4)(normal.z, normal.x, normal.y, 0.f);
		real velocityN = dot(velocity, normal);
		real velocityTA = dot(velocity, tangentA);
		real velocityTB = dot(velocity, tangentB);
	
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

		//eigenvectors

		//specify transposed
		real16 eigenvectorsA;
		real16 eigenvectorsB;
		real16 eigenvectorsC;
		real16 eigenvectorsD;
		
		//min col 
		eigenvectorsA.s0 = 1.f;
		eigenvectorsA.s8 = velocity.x - speedOfSound * normal.x;
		eigenvectorsB.s0 = velocity.y - speedOfSound * normal.y;
		eigenvectorsB.s8 = velocity.z - speedOfSound * normal.z;
		eigenvectorsC.s0 = enthalpyTotal - speedOfSound * velocityN;
		eigenvectorsC.s8 = 0.f;
		eigenvectorsD.s0 = 0.f;
		eigenvectorsD.s8 = 0.f;
		//mid col (normal)
		eigenvectorsA.s1 = 1.f;
		eigenvectorsA.s9 = velocity.x;
		eigenvectorsB.s1 = velocity.y;
		eigenvectorsB.s9 = velocity.z;
		eigenvectorsC.s1 = .5f * velocitySq;
		eigenvectorsC.s9 = 0.f;
		eigenvectorsD.s1 = 0.f;
		eigenvectorsD.s9 = 0.f;
		//mid col (tangent A)
		eigenvectorsA.s2 = 0.f;
		eigenvectorsA.sA = tangentA.x;
		eigenvectorsB.s2 = tangentA.y;
		eigenvectorsB.sA = tangentA.y;
		eigenvectorsC.s2 = velocityTA;
		eigenvectorsC.sA = 0.f;
		eigenvectorsD.s2 = 0.f;
		eigenvectorsD.sA = 0.f;
		//mid col (tangent B)
		eigenvectorsA.s3 = 0.f;
		eigenvectorsA.sB = tangentB.x;
		eigenvectorsB.s3 = tangentB.y;
		eigenvectorsB.sB = tangentB.y;
		eigenvectorsC.s3 = velocityTB;
		eigenvectorsC.sB = 0.f;
		eigenvectorsD.s3 = 0.f;
		eigenvectorsD.sB = 0.f;	
		//max col 
		eigenvectorsA.s4 = 1.f;
		eigenvectorsA.sC = velocity.x + speedOfSound * normal.x;
		eigenvectorsB.s4 = velocity.y + speedOfSound * normal.y;
		eigenvectorsB.sC = velocity.z + speedOfSound * normal.z;
		eigenvectorsC.s4 = enthalpyTotal + speedOfSound * velocityN;
		eigenvectorsC.sC = 0.f;
		eigenvectorsD.s4 = 0.f;
		eigenvectorsD.sC = 0.f;	

		eigenvectorsBuffer[0 + 4 * interfaceIndex] = eigenvectorsA;
		eigenvectorsBuffer[1 + 4 * interfaceIndex] = eigenvectorsB;
		eigenvectorsBuffer[2 + 4 * interfaceIndex] = eigenvectorsC;
		eigenvectorsBuffer[3 + 4 * interfaceIndex] = eigenvectorsD;

		//calculate eigenvector inverses ... 
		real invDenom = .5f / (speedOfSound * speedOfSound);
		eigenvectorsInverseBuffer[0 + 4 * interfaceIndex] = (real16)( 
		//min row
			(.5f * (GAMMA - 1.f) * velocitySq + speedOfSound * velocityN) * invDenom,
			-(normal.x * speedOfSound + (GAMMA - 1.f) * velocity.x) * invDenom,
			-(normal.y * speedOfSound + (GAMMA - 1.f) * velocity.y) * invDenom,
			-(normal.z * speedOfSound + (GAMMA - 1.f) * velocity.z) * invDenom,
			(GAMMA - 1.f) * invDenom,
			0.f,
			0.f, 
			0.f,
		//mid normal row
			1.f - (GAMMA - 1.f) * velocitySq * invDenom,
			(GAMMA - 1.f) * velocity.x * 2.f * invDenom,
			(GAMMA - 1.f) * velocity.y * 2.f * invDenom,
			(GAMMA - 1.f) * velocity.z * 2.f * invDenom,
			-(GAMMA - 1.f) * 2.f * invDenom,
			0.f,
			0.f,
			0.f);
		eigenvectorsInverseBuffer[1 + 4 * interfaceIndex] = (real16)(
		//mid tangent A row
			-velocityTA, 
			tangentA.x,
			tangentA.y,
			tangentA.z,
			0.f,
			0.f,
			0.f,
			0.f,
		//mid tangent B row
			-velocityTB,
			tangentB.x,
			tangentB.y,
			tangentB.z,
			0.f,
			0.f,
			0.f,
			0.f);
		eigenvectorsInverseBuffer[2 + 4 * interfaceIndex] = (real16)(
		//max row
			(.5f * (GAMMA - 1.f) * velocitySq - speedOfSound * velocityN) * invDenom,
			(normal.x * speedOfSound - (GAMMA - 1.f) * velocity.x) * invDenom,
			(normal.y * speedOfSound - (GAMMA - 1.f) * velocity.y) * invDenom,
			(normal.z * speedOfSound - (GAMMA - 1.f) * velocity.z) * invDenom,
			(GAMMA - 1.f) * invDenom,
			0.f,
			0.f,
			0.f,
			
			0.f,
			0.f,
			0.f,
			0.f,
			0.f,
			0.f,
			0.f,
			0.f);
		eigenvectorsInverseBuffer[3 + 4 * interfaceIndex] = (real16)(
			0.f,
			0.f,
			0.f,
			0.f,
			0.f,
			0.f,
			0.f,
			0.f,
			
			0.f,
			0.f,
			0.f,
			0.f,
			0.f,
			0.f,
			0.f,
			0.f);
#endif


#if 0	//calculate flux in x-axis and rotate into normal

		if (side == 1) {
			velocity = (real4)(velocity.y, -velocity.x, velocity.z, 0.f);	// -90' rotation to put the y axis contents into the x axis
		} else if (side == 2) {
			velocity = (real4)(velocity.z, velocity.y, -velocity.x, 0.f);	//-90' rotation to put the z axis in the x axis
		}

		//eigenvalues

		eigenvaluesBuffer[interfaceIndex] = (real8)(
			velocity.x - speedOfSound,
			velocity.x,
			velocity.x,
			velocity.x,
			velocity.x + speedOfSound,
			0.f,
			0.f,
			0.f);

		//eigenvectors

		//specify transposed
		real16 eigenvectorsA = (real16)(
		//min col 
			1.f,
			velocity.x - speedOfSound,
			velocity.y,
			velocity.z,
			enthalpyTotal - speedOfSound * velocity.x,
			0.f,
			0.f,
			0.f
		//mid col (normal)
			1.f,
			velocity.x,
			velocity.y,
			velocity.z,
			.5f * velocitySq,
			0.f,
			0.f,
			0.f);
		real16 eigenvectorsB = (real16)(
		//mid col (tangent A)
			0.f,
			0.f,
			1.f,
			0.f,
			velocity.y,
			0.f,
			0.f,
			0.f,
		//mid col (tangent B)
			0.f,
			0.f,
			0.f,
			1.f,
			velocity.z,
			0.f,
			0.f,
			0.f);
		real16 eigenvectorsC = (real16)(
		//max col 
			1.f,
			velocity.x + speedOfSound,
			velocity.y,
			velocity.z,
			enthalpyTotal + speedOfSound * velocity.x,
			0.f,
			0.f,
			0.f,
		//padding
			0.f,
			0.f,
			0.f,
			0.f,
			0.f,
			0.f,
			0.f,
			0.f);
		real16 eigenvectorsD = 0.f;

		//transpose and store
		eigenvectorsBuffer[0 + 4 * interfaceIndex] = (real16)(
			eigenvectorsA.s08, eigenvectorsB.s08, eigenvectorsC.s08, eigenvectorsD.s08,
			eigenvectorsA.s19, eigenvectorsB.s19, eigenvectorsC.s19, eigenvectorsD.s19);
		eigenvectorsBuffer[1 + 4 * interfaceIndex] = (real16)(
			eigenvectorsA.s2A, eigenvectorsB.s2A, eigenvectorsC.s2A, eigenvectorsD.s2A,
			eigenvectorsA.s3B, eigenvectorsB.s3B, eigenvectorsC.s3B, eigenvectorsD.s3B);
		eigenvectorsBuffer[2 + 4 * interfaceIndex] = (real16)(
			eigenvectorsA.s4C, eigenvectorsB.s4C, eigenvectorsC.s4C, eigenvectorsD.s4C,
			eigenvectorsA.s5D, eigenvectorsB.s5D, eigenvectorsC.s5D, eigenvectorsD.s5D);
		eigenvectorsBuffer[3 + 4 * interfaceIndex] = (real16)(
			eigenvectorsA.s6E, eigenvectorsB.s6E, eigenvectorsC.s6E, eigenvectorsD.s6E,
			eigenvectorsA.s7F, eigenvectorsB.s7F, eigenvectorsC.s7F, eigenvectorsD.s7F);

#error TODO
		//calculate eigenvector inverses ... 
		real invDenom = .5f / (speedOfSound * speedOfSound);
		real16 eigenvectorsInverse = (real16)( 
		//min row
			(.5f * (GAMMA - 1.f) * velocitySq + speedOfSound * velocity.x) * invDenom,
			-(speedOfSound + (GAMMA - 1.f) * velocity.x) * invDenom,
			-((GAMMA - 1.f) * velocity.y) * invDenom,
			(GAMMA - 1.f) * invDenom,
		//mid normal row
			1.f - (GAMMA - 1.f) * velocitySq * invDenom,
			(GAMMA - 1.f) * velocity.x * 2.f * invDenom,
			(GAMMA - 1.f) * velocity.y * 2.f * invDenom,
			-(GAMMA - 1.f) * 2.f * invDenom,
		//mid tangent row
			-velocity.y, 
			0.f,
			1.f,
			0.f,
		//max row
			(.5f * (GAMMA - 1.f) * velocitySq - speedOfSound * velocity.x) * invDenom,
			(speedOfSound - (GAMMA - 1.f) * velocity.x) * invDenom,
			-(GAMMA - 1.f) * velocity.y * invDenom,
			(GAMMA - 1.f) * invDenom);

		if (side == 1) {
			//-90' rotation applied to the LHS of incoming velocity vectors, to move their y axis into the x axis
			// is equivalent of a -90' rotation applied to the RHS of the flux jacobian A
			// and A = Q V Q-1 for Q = the right eigenvectors and Q-1 the left eigenvectors
			// so a -90' rotation applied to the RHS of A is a +90' rotation applied to the RHS of Q-1 the left eigenvectors
			//and while a rotation applied to the LHS of a vector rotates the elements of its column vectors, a rotation applied to the RHS rotates the elements of its row vectors 
			//each row's y <- x, x <- -y
			eigenvectorsInverse.s159D26AE = (real8)(-eigenvectorsInverse.s26AE, eigenvectorsInverse.s159D);
	
			//a -90' rotation applied to the RHS of A must be corrected with a 90' rotation on the LHS of A
			//this rotates the elements of the column vectors by 90'
			//each column's x <- y, y <- -x
			eigenvectors.s456789AB = (real8)(-eigenvectors.s89AB, eigenvectors.s4567);
		}
		
		eigenvectorsBuffer[4 * interfaceIndex] = eigenvectors;
		eigenvectorsInverseBuffer[4 * interfaceIndex] = eigenvectorsInverse;
#endif
	}
}


