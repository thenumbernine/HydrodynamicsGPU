#include "HydroGPU/Shared/Common3D.h"

real8 matmul(real16 ma, real16 mb, real8 v);
real8 slopeLimiter(real8 r);

//5x5 mat mul
// the 25 matrix entries are stored in ma:mb
real8 matmul(real16 ma, real16 mb, real8 v) {
	real8 result;
	result.s0123 = (real4)(
		dot(ma.s0123, v.s0123) + ma.s4 * v.s4,
		dot(ma.s5678, v.s0123) + ma.s9 * v.s4,
		dot(ma.sABCD, v.s0123) + ma.sE * v.s4,
		ma.sF * v.s0 + dot(mb.s0123, v.s1234));
	result.s4 = 
		dot(mb.s4567, v.s0123) + mb.s8 * v.s4;
	return result;
}

//crashing when compiled ...
__kernel void calcEigenBasis(
	__global real8* eigenvaluesBuffer,
	__global real16* eigenvectorsBuffer,
	__global real16* eigenvectorsInverseBuffer,
	const __global real8* stateBuffer,
	const __global real* gravityPotentialBuffer,
	int3 size)
{
#if 0
	int3 i = (int3)(get_global_id(0), get_global_id(1), get_global_id(2));
	if (i.x < 2 || i.y < 2 || i.z < 2 || i.x >= size.x - 1 || i.y >= size.y - 1 || i.z >= size.z - 1) return;
	int index = INDEXV(i);

	for (int side = 0; side < 3; ++side) {	
		int3 iPrev = i;
		--iPrev[side];
		int indexPrev = INDEXV(iPrev);

		int interfaceIndex = side + 3 * index;

		real8 stateL = stateBuffer[indexPrev];
		real8 stateR = stateBuffer[index];

		real densityL = stateL.s0;
		real invDensityL = 1.f / densityL;
		real3 velocityL = stateL.s123 * invDensityL;
		real energyTotalL = stateL.s4 * invDensityL;

		real densityR = stateR.s0;
		real invDensityR = 1.f / densityR;
		real3 velocityR = stateR.s123 * invDensityR;
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
		real3 velocity = (weightL * velocityL + weightR * velocityR) * roeWeightNormalization;
		real enthalpyTotal = (weightL * enthalpyTotalL + weightR * enthalpyTotalR) * roeWeightNormalization;
		
		real velocitySq = dot(velocity, velocity);
		real speedOfSound = sqrt((GAMMA - 1.f) * (enthalpyTotal - .5f * velocitySq));
		
#if 1	//calculate flux based on normal

		real3 normal = (real3)(0.f, 0.f, 0.f);
		normal[side] = 1;

		real3 tangentA = (real3)(normal.y, normal.z, normal.x);
		real3 tangentB = (real3)(normal.z, normal.x, normal.y);
		real velocityN = dot(velocity, normal);
		real velocityTA = dot(velocity, tangentA);
		real velocityTB = dot(velocity, tangentB);
	
		//eigenvalues

		real8 eigenvalues;
		eigenvalues.s0123 = (real4)(
			velocityN - speedOfSound,
			velocityN,
			velocityN,
			velocityN);
		eigenvalues.s4 = velocityN + speedOfSound;
		eigenvaluesBuffer[interfaceIndex] = eigenvalues;

		//eigenvectors

		//specify transposed
		real16 eigenvectorsA, eigenvectorsB;
		//min col 
		eigenvectorsA[0] = 1.f;
		eigenvectorsA[5] = velocity.x - speedOfSound * normal.x;
		eigenvectorsA[10] = velocity.y - speedOfSound * normal.y;
		eigenvectorsA[15] = velocity.z - speedOfSound * normal.z;
		eigenvectorsB[20-16] = enthalpyTotal - speedOfSound * velocityN;
		//mid col (normal)
		eigenvectorsA[1] = 1.f;
		eigenvectorsA[6] = velocity.x;
		eigenvectorsA[11] = velocity.y;
		eigenvectorsB[16-16] = velocity.z;
		eigenvectorsB[21-16] = .5f * velocitySq;
		//mid col (tangent A)
		eigenvectorsA[2] = 0.f;
		eigenvectorsA[7] = tangentA.x;
		eigenvectorsA[12] = tangentA.y;
		eigenvectorsB[17-16] = tangentA.z;
		eigenvectorsB[22-16] = velocityTA;
		//mid col (tangent B)
		eigenvectorsA[3] = 0.f;
		eigenvectorsA[8] = tangentB.x;
		eigenvectorsA[13] = tangentB.y;
		eigenvectorsB[18-16] = tangentB.z;
		eigenvectorsB[23-16] = velocityTB;
		//max col 
		eigenvectorsA[4] = 1.f;
		eigenvectorsA[9] = velocity.x + speedOfSound * normal.x;
		eigenvectorsA[14] = velocity.y + speedOfSound * normal.y;
		eigenvectorsB[19-16] = velocity.z + speedOfSound * normal.z;
		eigenvectorsB[24-16] = enthalpyTotal + speedOfSound * velocityN;

		eigenvectorsBuffer[0 + 2 * interfaceIndex] = eigenvectorsA;
		eigenvectorsBuffer[1 + 2 * interfaceIndex] = eigenvectorsB;


		//calculate eigenvector inverses ... 
		real invDenom = .5f / (speedOfSound * speedOfSound);
		real16 eigenvectorsInverseA, eigenvectorsInverseB;
		eigenvectorsInverseA = (real16)( 
		//min row
			(.5f * (GAMMA - 1.f) * velocitySq + speedOfSound * velocityN) * invDenom,
			-(normal.x * speedOfSound + (GAMMA - 1.f) * velocity.x) * invDenom,
			-(normal.y * speedOfSound + (GAMMA - 1.f) * velocity.y) * invDenom,
			-(normal.z * speedOfSound + (GAMMA - 1.f) * velocity.z) * invDenom,
			(GAMMA - 1.f) * invDenom,
		//mid normal row
			1.f - (GAMMA - 1.f) * velocitySq * invDenom,
			(GAMMA - 1.f) * velocity.x * 2.f * invDenom,
			(GAMMA - 1.f) * velocity.y * 2.f * invDenom,
			(GAMMA - 1.f) * velocity.z * 2.f * invDenom,
			-(GAMMA - 1.f) * 2.f * invDenom,
		//mid tangent A row
			-velocityTA,
			tangentA.x,
			tangentA.y,
			tangentA.z,
			0.f,
		//mid tangent B row
			-velocityTB);
		eigenvectorsInverseB.s01234567 = (real8)(
			tangentB.x,
			tangentB.y,
			tangentB.z,
			0.f,
		//max row
			(.5f * (GAMMA - 1.f) * velocitySq - speedOfSound * velocityN) * invDenom,
			(normal.x * speedOfSound - (GAMMA - 1.f) * velocity.x) * invDenom,
			(normal.y * speedOfSound - (GAMMA - 1.f) * velocity.y) * invDenom,
			(normal.z * speedOfSound - (GAMMA - 1.f) * velocity.z) * invDenom);
		eigenvectorsB.s8 =
			(GAMMA - 1.f) * invDenom;

		eigenvectorsInverseBuffer[0 + 2 * interfaceIndex] = eigenvectorsInverseA;
		eigenvectorsInverseBuffer[1 + 2 * interfaceIndex] = eigenvectorsInverseB;
#endif



#if 0	//calculate flux in x-axis and rotate into normal
		
		// -90' rotation to put the [side] axis contents into the x axis
		if (side == 1) {
			velocity.xy = (real2)(velocity.y, -velocity.x);
		} else if (side == 2) {
			velocity.xz = (real2)(velocity.z, -velocity.x);
		}

		//eigenvalues

		eigenvaluesBuffer[interfaceIndex].s0123 =  (real4)(
			velocity.x - speedOfSound,
			velocity.x,
			velocity.x,
			velocity.x + speedOfSound);

		//eigenvectors

		//specify transposed
		real16 eigenvectors = (real16)(
		//min col 
			1.f,
			velocity.x - speedOfSound,
			velocity.y,
			enthalpyTotal - speedOfSound * velocity.x,
		//mid col (normal)
			1.f,
			velocity.x,
			velocity.y,
			.5f * velocitySq,
		//mid col (tangent)
			0.f,
			0.f,
			1.f,
			velocity.y,
		//max col 
			1.f,
			velocity.x + speedOfSound,
			velocity.y,
			enthalpyTotal + speedOfSound * velocity.x);

		//transpose and store
		eigenvectors = (real16)(
			eigenvectors.s048C,
			eigenvectors.s159D,
			eigenvectors.s26AE,
			eigenvectors.s37BF);

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
		
		eigenvectorsBuffer[interfaceIndex] = eigenvectors;
		eigenvectorsInverseBuffer[interfaceIndex] = eigenvectorsInverse;
#endif
	}
#endif
}

__kernel void calcCFL(
	__global real* cflBuffer,
	const __global real8* eigenvaluesBuffer,
	int3 size,
	real3 dx,
	real cfl)
{
	int3 i = (int3)(get_global_id(0), get_global_id(1), get_global_id(2));
	int index = INDEXV(i);
	if (i.x < 2 || i.y < 2 || i.z < 2 || i.x >= size.x - 2 || i.y >= size.y - 2 || i.z >= size.z - 2) {
		cflBuffer[index] = INFINITY;
		return;
	}

	real3 dum;
	for (int side = 0; side < 3; ++side) {
		int3 iNext = i;
		++iNext[side];
		int indexNext = INDEXV(iNext);
		
		real8 eigenvaluesL = eigenvaluesBuffer[side + 3 * index];
		real8 eigenvaluesR = eigenvaluesBuffer[side + 3 * indexNext];
		
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

		dum[side] = dx[side] / (maxLambda - minLambda);
	}
		
	cflBuffer[index] = cfl * min(dum.x, min(dum.y, dum.z));
}

__kernel void calcDeltaQTilde(
	__global real8* deltaQTildeBuffer,
	const __global real16* eigenvectorsInverseBuffer,
	const __global real8* stateBuffer,
	int3 size,
	real2 dx)
{
	int3 i = (int3)(get_global_id(0), get_global_id(1), get_global_id(2));
	if (i.x < 2 || i.y < 2 || i.z < 2 || i.x >= size.x - 1 || i.y >= size.y - 1 || i.z >= size.z - 1) return;
	int index = INDEXV(i);

	for (int side = 0; side < 3; ++side) {
		int3 iPrev = i;
		--iPrev[side];
		int indexPrev = INDEXV(iPrev);
				
		real8 stateL = stateBuffer[indexPrev];
		real8 stateR = stateBuffer[index];

		int interfaceIndex = side + 3 * index;
		
		real8 deltaQ = stateR - stateL;
		deltaQTildeBuffer[interfaceIndex] = matmul(
			eigenvectorsInverseBuffer[0 + 2 * interfaceIndex], 
			eigenvectorsInverseBuffer[1 + 2 * interfaceIndex], 
			deltaQ);
	}
}

real8 slopeLimiter(real8 r) {
	//donor cell
	//return (real8)(0.f, 0.f, 0.f, 0.f);
	//superbee
	return max(0.f, max(min(1.f, 2.f * r), min(2.f, r)));
}

//crashing on compile...
__kernel void calcFlux(
	__global real8* fluxBuffer,
	const __global real8* stateBuffer,
	const __global real8* eigenvaluesBuffer,
	const __global real16* eigenvectorsBuffer,
	const __global real16* eigenvectorsInverseBuffer,
	const __global real8* deltaQTildeBuffer,
	int3 size,
	real3 dx,
	__global real *dt)
{
#if 0
	real3 dt_dx = *dt / dx;
	
	int3 i = (int3)(get_global_id(0), get_global_id(1), get_global_id(2));
	if (i.x < 2 || i.y < 2 || i.z < 2 || i.x >= size.x - 1 || i.y >= size.y - 1 || i.z >= size.z - 1) return;
	int index = INDEXV(i);
	
	for (int side = 0; side < 3; ++side) {	
		int3 iPrev = i;
		--iPrev[side];
		int indexPrev = iPrev.x + size.x * iPrev.y;

		int3 iNext = i;
		++iNext[side];
		int indexNext = iNext.x + size.x * iNext.y;
	
		real8 stateL = stateBuffer[indexPrev];
		real8 stateR = stateBuffer[index];

		//int interfaceLIndex = side + 3 * indexPrev;
		int interfaceIndex = side + 3 * index;
		//int interfaceRIndex = side + 3 * indexNext;

		//real8 deltaQTildeL = deltaQTildeBuffer[interfaceLIndex];
		real8 deltaQTilde = deltaQTildeBuffer[interfaceIndex];
		//real8 deltaQTildeR = deltaQTildeBuffer[interfaceRIndex];

		real8 eigenvalues = eigenvaluesBuffer[interfaceIndex];
		real16 eigenvectorsA = eigenvectorsBuffer[0 + 2 * interfaceIndex];
		real16 eigenvectorsB = eigenvectorsBuffer[1 + 2 * interfaceIndex];
		real16 eigenvectorsInverseA = eigenvectorsInverseBuffer[0 + 2 * interfaceIndex];
		real16 eigenvectorsInverseB = eigenvectorsInverseBuffer[1 + 2 * interfaceIndex];

		real8 theta = step(0.f, eigenvalues) * 2.f - 1.f;
		//real8 rTilde = mix(deltaQTildeR, deltaQTildeL, theta * .5f + .5f) / deltaQTilde;
		real8 qAvg = (stateR + stateL) * .5f;
		real8 fluxAvgTilde = matmul(eigenvectorsInverseA, eigenvectorsInverseB, qAvg) * eigenvalues;
		//real8 phi = slopeLimiter(rTilde);
		//real8 epsilon = eigenvalues * dt_dx[side];
		real8 deltaFluxTilde = eigenvalues * deltaQTilde;
		real8 fluxTilde = fluxAvgTilde;
		
		//combined, delta factored out:
		//fluxTilde -= .5f * deltaFluxTilde * (theta + .5f * phi * (epsilon - theta));
		
		//split apart:
		fluxTilde -= .5f * deltaFluxTilde * theta;
		//fluxTilde -= .25f * deltaFluxTilde * phi * (epsilon - theta);
		
		fluxBuffer[side + 3 * index] = matmul(eigenvectorsA, eigenvectorsB, fluxTilde);
	}
#endif
}

__kernel void integrateFlux(
	__global real8* stateBuffer,
	const __global real8* fluxBuffer,
	int3 size,
	real3 dx,
	const __global real* dt)
{
	real3 dt_dx = *dt / dx;
	
	int3 i = (int3)(get_global_id(0), get_global_id(1), get_global_id(2));
	if (i.x < 2 || i.y < 2 || i.z < 2 || i.x >= size.x - 2 || i.y >= size.y - 2 || i.z >= size.z - 2) return;
	int index = INDEXV(i);
	
	for (int side = 0; side < 3; ++side) {	
		int3 iNext = i;
		++iNext[side];
		int indexNext = INDEXV(iNext);
		
		real8 fluxL = fluxBuffer[side + 3 * index];
		real8 fluxR = fluxBuffer[side + 3 * indexNext];

		real8 df = fluxR - fluxL;
		stateBuffer[index] -= df * dt_dx[side];
	}
}

