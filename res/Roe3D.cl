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
	__global real4* eigenvectorsBuffer,
	__global real4* eigenvectorsInverseBuffer,
	const __global real8* stateBuffer,
	const __global real* gravityPotentialBuffer,
	int3 size)
{
	int3 i = (int3)(get_global_id(0), get_global_id(1), get_global_id(2));
	if (i.x < 2 || i.y < 2 || i.z < 2 || i.x >= size.x - 1 || i.y >= size.y - 1 || i.z >= size.z - 1) return;
	int index = INDEXV(i);

	for (int side = 0; side < DIM; ++side) {	
		int3 iPrev = i;
		--iPrev[side];
		int indexPrev = INDEXV(iPrev);

		int interfaceIndex = side + DIM * index;

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
		eigenvaluesBuffer[interfaceIndex] = (real8)(
			velocityN - speedOfSound,
			velocityN,
			velocityN,
			velocityN,
			velocityN + speedOfSound,
			0.f, 0.f, 0.f);

		//eigenvectors

		//specify transposed
		real16 eigenvectorsA, eigenvectorsB;
		//min col 
		eigenvectorsA.s0 = 1.f;
		eigenvectorsA.s5 = velocity.x - speedOfSound * normal.x;
		eigenvectorsA.sA = velocity.y - speedOfSound * normal.y;
		eigenvectorsA.sF = velocity.z - speedOfSound * normal.z;
		eigenvectorsB.s4 = enthalpyTotal - speedOfSound * velocityN;
		//mid col (normal)
		eigenvectorsA.s1 = 1.f;
		eigenvectorsA.s6 = velocity.x;
		eigenvectorsA.sB = velocity.y;
		eigenvectorsB.s0 = velocity.z;
		eigenvectorsB.s5 = .5f * velocitySq;
		//mid col (tangent A)
		eigenvectorsA.s2 = 0.f;
		eigenvectorsA.s7 = tangentA.x;
		eigenvectorsA.sC = tangentA.y;
		eigenvectorsB.s1 = tangentA.z;
		eigenvectorsB.s6 = velocityTA;
		//mid col (tangent B)
		eigenvectorsA.s3 = 0.f;
		eigenvectorsA.s8 = tangentB.x;
		eigenvectorsA.sC = tangentB.y;
		eigenvectorsB.s2 = tangentB.z;
		eigenvectorsB.s7 = velocityTB;
		//max col 
		eigenvectorsA.s4 = 1.f;
		eigenvectorsA.s9 = velocity.x + speedOfSound * normal.x;
		eigenvectorsA.sD = velocity.y + speedOfSound * normal.y;
		eigenvectorsB.s3 = velocity.z + speedOfSound * normal.z;
		eigenvectorsB.s8 = enthalpyTotal + speedOfSound * velocityN;

		eigenvectorsBuffer[0 + 8 * interfaceIndex] = eigenvectorsA.s0123;
		eigenvectorsBuffer[1 + 8 * interfaceIndex] = eigenvectorsA.s4567;
		eigenvectorsBuffer[2 + 8 * interfaceIndex] = eigenvectorsA.s89AB;
		eigenvectorsBuffer[3 + 8 * interfaceIndex] = eigenvectorsA.sCDEF;
		eigenvectorsBuffer[4 + 8 * interfaceIndex] = eigenvectorsB.s0123;
		eigenvectorsBuffer[5 + 8 * interfaceIndex] = eigenvectorsB.s4567;
		eigenvectorsBuffer[6 + 8 * interfaceIndex] = eigenvectorsB.s89AB;
		eigenvectorsBuffer[7 + 8 * interfaceIndex] = eigenvectorsB.sCDEF;

		//calculate eigenvector inverses ... 
		real invDenom = .5f / (speedOfSound * speedOfSound);
		real16 eigenvectorsInverseA;
		real16 eigenvectorsInverseB;
		//min row
		eigenvectorsInverseA.s0 = (.5f * (GAMMA - 1.f) * velocitySq + speedOfSound * velocityN) * invDenom;
		eigenvectorsInverseA.s1 = -(normal.x * speedOfSound + (GAMMA - 1.f) * velocity.x) * invDenom;
		eigenvectorsInverseA.s2 = -(normal.y * speedOfSound + (GAMMA - 1.f) * velocity.y) * invDenom;
		eigenvectorsInverseA.s3 = -(normal.z * speedOfSound + (GAMMA - 1.f) * velocity.z) * invDenom;
		eigenvectorsInverseA.s4 = (GAMMA - 1.f) * invDenom;
		//mid normal row
		eigenvectorsInverseA.s5 = 1.f - (GAMMA - 1.f) * velocitySq * invDenom;
		eigenvectorsInverseA.s6 = (GAMMA - 1.f) * velocity.x * 2.f * invDenom;
		eigenvectorsInverseA.s7 = (GAMMA - 1.f) * velocity.y * 2.f * invDenom;
		eigenvectorsInverseA.s8 = (GAMMA - 1.f) * velocity.z * 2.f * invDenom;
		eigenvectorsInverseA.s9 = -(GAMMA - 1.f) * 2.f * invDenom;
		//mid tangent A row
		eigenvectorsInverseA.sA = -velocityTA;
		eigenvectorsInverseA.sB = tangentA.x;
		eigenvectorsInverseA.sC = tangentA.y;
		eigenvectorsInverseA.sD = tangentA.z;
		eigenvectorsInverseA.sE = 0.f;
		//mid tangent B row
		eigenvectorsInverseA.sF = -velocityTB;
		eigenvectorsInverseB.s0 = tangentB.x;
		eigenvectorsInverseB.s1 = tangentB.y;
		eigenvectorsInverseB.s2 = tangentB.z;
		eigenvectorsInverseB.s3 = 0.f;
		//max row
		eigenvectorsInverseB.s4 = (.5f * (GAMMA - 1.f) * velocitySq - speedOfSound * velocityN) * invDenom;
		eigenvectorsInverseB.s5 = (normal.x * speedOfSound - (GAMMA - 1.f) * velocity.x) * invDenom;
		eigenvectorsInverseB.s6 = (normal.y * speedOfSound - (GAMMA - 1.f) * velocity.y) * invDenom;
		eigenvectorsInverseB.s7 = (normal.z * speedOfSound - (GAMMA - 1.f) * velocity.z) * invDenom;
		eigenvectorsInverseB.s8 = (GAMMA - 1.f) * invDenom;
		eigenvectorsInverseB.s9 = 0.f;
		eigenvectorsInverseB.sA = 0.f;
		eigenvectorsInverseB.sB = 0.f;
		eigenvectorsInverseB.sC = 0.f;
		eigenvectorsInverseB.sD = 0.f;
		eigenvectorsInverseB.sE = 0.f;
		eigenvectorsInverseB.sF = 0.f;

		eigenvectorsInverseBuffer[0 + 8 * interfaceIndex] = eigenvectorsInverseA.s0123;
#if 0
		//writing to these upper 4 causes it to crash
		eigenvectorsInverseBuffer[0 + 2 * interfaceIndex].s4567 = eigenvectorsInverseA.s4567;
		eigenvectorsInverseBuffer[1 + 2 * interfaceIndex] = eigenvectorsInverseB;
#endif
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

