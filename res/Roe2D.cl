#include "HydroGPU/Shared/Common2D.h"

real8 matmul(real16 ma, real16 mb, real16 mc, real16 md, real8 v);
real mat44det(real16 m);
real16 mat44inv(real16 m);
real8 slopeLimiter(real8 r);

real8 matmul(real16 ma, real16 mb, real16 mc, real16 md, real8 v) {
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

real mat44det(real16 m) {
	return m[0+4*3]*m[1+4*2]*m[2+4*1]*m[3+4*0] - m[0+4*2]*m[1+4*3]*m[2+4*1]*m[3+4*0] - m[0+4*3]*m[1+4*1]*m[2+4*2]*m[3+4*0] + m[0+4*1]*m[1+4*3]*m[2+4*2]*m[3+4*0]+
		m[0+4*2]*m[1+4*1]*m[2+4*3]*m[3+4*0] - m[0+4*1]*m[1+4*2]*m[2+4*3]*m[3+4*0] - m[0+4*3]*m[1+4*2]*m[2+4*0]*m[3+4*1] + m[0+4*2]*m[1+4*3]*m[2+4*0]*m[3+4*1]+
		m[0+4*3]*m[1+4*0]*m[2+4*2]*m[3+4*1] - m[0+4*0]*m[1+4*3]*m[2+4*2]*m[3+4*1] - m[0+4*2]*m[1+4*0]*m[2+4*3]*m[3+4*1] + m[0+4*0]*m[1+4*2]*m[2+4*3]*m[3+4*1]+
		m[0+4*3]*m[1+4*1]*m[2+4*0]*m[3+4*2] - m[0+4*1]*m[1+4*3]*m[2+4*0]*m[3+4*2] - m[0+4*3]*m[1+4*0]*m[2+4*1]*m[3+4*2] + m[0+4*0]*m[1+4*3]*m[2+4*1]*m[3+4*2]+
		m[0+4*1]*m[1+4*0]*m[2+4*3]*m[3+4*2] - m[0+4*0]*m[1+4*1]*m[2+4*3]*m[3+4*2] - m[0+4*2]*m[1+4*1]*m[2+4*0]*m[3+4*3] + m[0+4*1]*m[1+4*2]*m[2+4*0]*m[3+4*3]+
		m[0+4*2]*m[1+4*0]*m[2+4*1]*m[3+4*3] - m[0+4*0]*m[1+4*2]*m[2+4*1]*m[3+4*3] - m[0+4*1]*m[1+4*0]*m[2+4*2]*m[3+4*3] + m[0+4*0]*m[1+4*1]*m[2+4*2]*m[3+4*3];
}

real16 mat44inv(real16 m) {
	real16 r;
	r[0+4*0] = m[1+4*2]*m[2+4*3]*m[3+4*1] - m[1+4*3]*m[2+4*2]*m[3+4*1] + m[1+4*3]*m[2+4*1]*m[3+4*2] - m[1+4*1]*m[2+4*3]*m[3+4*2] - m[1+4*2]*m[2+4*1]*m[3+4*3] + m[1+4*1]*m[2+4*2]*m[3+4*3];
	r[0+4*1] = m[0+4*3]*m[2+4*2]*m[3+4*1] - m[0+4*2]*m[2+4*3]*m[3+4*1] - m[0+4*3]*m[2+4*1]*m[3+4*2] + m[0+4*1]*m[2+4*3]*m[3+4*2] + m[0+4*2]*m[2+4*1]*m[3+4*3] - m[0+4*1]*m[2+4*2]*m[3+4*3];
	r[0+4*2] = m[0+4*2]*m[1+4*3]*m[3+4*1] - m[0+4*3]*m[1+4*2]*m[3+4*1] + m[0+4*3]*m[1+4*1]*m[3+4*2] - m[0+4*1]*m[1+4*3]*m[3+4*2] - m[0+4*2]*m[1+4*1]*m[3+4*3] + m[0+4*1]*m[1+4*2]*m[3+4*3];
	r[0+4*3] = m[0+4*3]*m[1+4*2]*m[2+4*1] - m[0+4*2]*m[1+4*3]*m[2+4*1] - m[0+4*3]*m[1+4*1]*m[2+4*2] + m[0+4*1]*m[1+4*3]*m[2+4*2] + m[0+4*2]*m[1+4*1]*m[2+4*3] - m[0+4*1]*m[1+4*2]*m[2+4*3];
	r[1+4*0] = m[1+4*3]*m[2+4*2]*m[3+4*0] - m[1+4*2]*m[2+4*3]*m[3+4*0] - m[1+4*3]*m[2+4*0]*m[3+4*2] + m[1+4*0]*m[2+4*3]*m[3+4*2] + m[1+4*2]*m[2+4*0]*m[3+4*3] - m[1+4*0]*m[2+4*2]*m[3+4*3];
	r[1+4*1] = m[0+4*2]*m[2+4*3]*m[3+4*0] - m[0+4*3]*m[2+4*2]*m[3+4*0] + m[0+4*3]*m[2+4*0]*m[3+4*2] - m[0+4*0]*m[2+4*3]*m[3+4*2] - m[0+4*2]*m[2+4*0]*m[3+4*3] + m[0+4*0]*m[2+4*2]*m[3+4*3];
	r[1+4*2] = m[0+4*3]*m[1+4*2]*m[3+4*0] - m[0+4*2]*m[1+4*3]*m[3+4*0] - m[0+4*3]*m[1+4*0]*m[3+4*2] + m[0+4*0]*m[1+4*3]*m[3+4*2] + m[0+4*2]*m[1+4*0]*m[3+4*3] - m[0+4*0]*m[1+4*2]*m[3+4*3];
	r[1+4*3] = m[0+4*2]*m[1+4*3]*m[2+4*0] - m[0+4*3]*m[1+4*2]*m[2+4*0] + m[0+4*3]*m[1+4*0]*m[2+4*2] - m[0+4*0]*m[1+4*3]*m[2+4*2] - m[0+4*2]*m[1+4*0]*m[2+4*3] + m[0+4*0]*m[1+4*2]*m[2+4*3];
	r[2+4*0] = m[1+4*1]*m[2+4*3]*m[3+4*0] - m[1+4*3]*m[2+4*1]*m[3+4*0] + m[1+4*3]*m[2+4*0]*m[3+4*1] - m[1+4*0]*m[2+4*3]*m[3+4*1] - m[1+4*1]*m[2+4*0]*m[3+4*3] + m[1+4*0]*m[2+4*1]*m[3+4*3];
	r[2+4*1] = m[0+4*3]*m[2+4*1]*m[3+4*0] - m[0+4*1]*m[2+4*3]*m[3+4*0] - m[0+4*3]*m[2+4*0]*m[3+4*1] + m[0+4*0]*m[2+4*3]*m[3+4*1] + m[0+4*1]*m[2+4*0]*m[3+4*3] - m[0+4*0]*m[2+4*1]*m[3+4*3];
	r[2+4*2] = m[0+4*1]*m[1+4*3]*m[3+4*0] - m[0+4*3]*m[1+4*1]*m[3+4*0] + m[0+4*3]*m[1+4*0]*m[3+4*1] - m[0+4*0]*m[1+4*3]*m[3+4*1] - m[0+4*1]*m[1+4*0]*m[3+4*3] + m[0+4*0]*m[1+4*1]*m[3+4*3];
	r[2+4*3] = m[0+4*3]*m[1+4*1]*m[2+4*0] - m[0+4*1]*m[1+4*3]*m[2+4*0] - m[0+4*3]*m[1+4*0]*m[2+4*1] + m[0+4*0]*m[1+4*3]*m[2+4*1] + m[0+4*1]*m[1+4*0]*m[2+4*3] - m[0+4*0]*m[1+4*1]*m[2+4*3];
	r[3+4*0] = m[1+4*2]*m[2+4*1]*m[3+4*0] - m[1+4*1]*m[2+4*2]*m[3+4*0] - m[1+4*2]*m[2+4*0]*m[3+4*1] + m[1+4*0]*m[2+4*2]*m[3+4*1] + m[1+4*1]*m[2+4*0]*m[3+4*2] - m[1+4*0]*m[2+4*1]*m[3+4*2];
	r[3+4*1] = m[0+4*1]*m[2+4*2]*m[3+4*0] - m[0+4*2]*m[2+4*1]*m[3+4*0] + m[0+4*2]*m[2+4*0]*m[3+4*1] - m[0+4*0]*m[2+4*2]*m[3+4*1] - m[0+4*1]*m[2+4*0]*m[3+4*2] + m[0+4*0]*m[2+4*1]*m[3+4*2];
	r[3+4*2] = m[0+4*2]*m[1+4*1]*m[3+4*0] - m[0+4*1]*m[1+4*2]*m[3+4*0] - m[0+4*2]*m[1+4*0]*m[3+4*1] + m[0+4*0]*m[1+4*2]*m[3+4*1] + m[0+4*1]*m[1+4*0]*m[3+4*2] - m[0+4*0]*m[1+4*1]*m[3+4*2];
	r[3+4*3] = m[0+4*1]*m[1+4*2]*m[2+4*0] - m[0+4*2]*m[1+4*1]*m[2+4*0] + m[0+4*2]*m[1+4*0]*m[2+4*1] - m[0+4*0]*m[1+4*2]*m[2+4*1] - m[0+4*1]*m[1+4*0]*m[2+4*2] + m[0+4*0]*m[1+4*1]*m[2+4*2];
	return r * (1.f / mat44det(m));
}

__kernel void calcEigenBasis(
	__global real8* eigenvaluesBuffer,
	__global real16* eigenvectorsBuffer,
	__global real16* eigenvectorsInverseBuffer,
	const __global real8* stateBuffer,
	const __global real* gravityPotentialBuffer,
	int2 size)
{
	int2 i = (int2)(get_global_id(0), get_global_id(1));
	if (i.x < 2 || i.x >= size.x - 1 
#if DIM > 1
		|| i.y < 2 || i.y >= size.y - 1
#endif
	) return;
	int index = INDEXV(i);

	for (int side = 0; side < DIM; ++side) {	
		int2 iPrev = i;
		--iPrev[side];
		int indexPrev = INDEXV(iPrev);

		int interfaceIndex = side + DIM * index;

		real8 stateL = stateBuffer[indexPrev];
		real8 stateR = stateBuffer[index];

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
	
#if 1	//calculate flux based on normal

		real2 normal = (real2)(0.f, 0.f);
		normal[side] = 1;
	
		real2 tangent = (real2)(-normal.y, normal.x);
		real velocityN = dot(velocity, normal);
		real velocityT = dot(velocity, tangent);
	
		//eigenvalues

		eigenvaluesBuffer[interfaceIndex].s0123 = (real4)(
			velocityN - speedOfSound,
			velocityN,
			velocityN,
			velocityN + speedOfSound);

		//eigenvectors

		//specify transposed
		real16 eigenvectorsA;
		real16 eigenvectorsB;
		//real16 eigenvectorsC;
		//real16 eigenvectorsD;
		
		//min col 
		eigenvectorsA.s0 = 1.f;
		eigenvectorsA.s8 = velocity.x - speedOfSound * normal.x;
		eigenvectorsB.s0 = velocity.y - speedOfSound * normal.y;
		eigenvectorsB.s8 = enthalpyTotal - speedOfSound * velocityN;
		//mid col (normal)
		eigenvectorsA.s1 = 1.f;
		eigenvectorsA.s9 = velocity.x;
		eigenvectorsB.s1 = velocity.y;
		eigenvectorsB.s9 = .5f * velocitySq;
		//mid col (tangent)
		eigenvectorsA.s2 = 0.f;
		eigenvectorsA.sA = tangent.x;
		eigenvectorsB.s2 = tangent.y;
		eigenvectorsB.sA = velocityT;
		//max col 
		eigenvectorsA.s3 = 1.f;
		eigenvectorsA.sB = velocity.x + speedOfSound * normal.x;
		eigenvectorsB.s3 = velocity.y + speedOfSound * normal.y;
		eigenvectorsB.sB = enthalpyTotal + speedOfSound * velocityN;

		//transpose and store
		eigenvectorsBuffer[0 + 4 * interfaceIndex] = eigenvectorsA;
		eigenvectorsBuffer[1 + 4 * interfaceIndex] = eigenvectorsB;
		//eigenvectorsBuffer[2 + 4 * interfaceIndex] = eigenvectorsC;
		//eigenvectorsBuffer[3 + 4 * interfaceIndex] = eigenvectorsD;

#if 1	//analytical eigenvalues.  getting worse results on double precision on my CPU-driven hydrodynamics program
		//calculate eigenvector inverses ... 
		real invDenom = .5f / (speedOfSound * speedOfSound);
		eigenvectorsInverseBuffer[0 + 4 * interfaceIndex] = (real16)( 
		//min row
			(.5f * (GAMMA - 1.f) * velocitySq + speedOfSound * velocityN) * invDenom,
			-(normal.x * speedOfSound + (GAMMA - 1.f) * velocity.x) * invDenom,
			-(normal.y * speedOfSound + (GAMMA - 1.f) * velocity.y) * invDenom,
			(GAMMA - 1.f) * invDenom,
			0.f, 0.f, 0.f, 0.f,
		//mid normal row
			1.f - (GAMMA - 1.f) * velocitySq * invDenom,
			(GAMMA - 1.f) * velocity.x * 2.f * invDenom,
			(GAMMA - 1.f) * velocity.y * 2.f * invDenom,
			-(GAMMA - 1.f) * 2.f * invDenom,
			0.f, 0.f, 0.f, 0.f);
		eigenvectorsInverseBuffer[1 + 4 * interfaceIndex] = (real16)(
		//mid tangent row
			-velocityT, 
			tangent.x,
			tangent.y,
			0.f,
			0.f, 0.f, 0.f, 0.f,
		//max row
			(.5f * (GAMMA - 1.f) * velocitySq - speedOfSound * velocityN) * invDenom,
			(normal.x * speedOfSound - (GAMMA - 1.f) * velocity.x) * invDenom,
			(normal.y * speedOfSound - (GAMMA - 1.f) * velocity.y) * invDenom,
			(GAMMA - 1.f) * invDenom,
			0.f, 0.f, 0.f, 0.f);
#endif
#if 0 //numerically solve for the inverse
		eigenvectorsInverseBuffer[4 * interfaceIndex] = mat44inv(eigenvectorsBuffer[4 * interfaceIndex]);
#endif
#endif


#if 0	//calculate flux in x-axis and rotate into normal

		if (side == 1) {
			velocity.xy = (real2)(velocity.y, -velocity.x);	// -90' rotation to put the y axis contents into the x axis
		}

		//eigenvalues

		eigenvaluesBuffer[interfaceIndex].s0123 = (real4)(
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

#if 1	//analytical eigenvalues.  getting worse results on double precision on my CPU-driven hydrodynamics program
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
#endif
#if 0 //numerically solve for the inverse
		real16 eigenvectorsInverse = mat44inv(eigenvectors);
#endif

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

__kernel void calcCFL(
	__global real* cflBuffer,
	const __global real8* eigenvaluesBuffer,
	int2 size,
	real2 dx,
	real cfl)
{
	int2 i = (int2)(get_global_id(0), get_global_id(1));
	int index = INDEXV(i);
	if (i.x < 2 || i.x >= size.x - 2 
#if DIM > 1
		|| i.y < 2 || i.y >= size.y - 2
#endif
	) {
		cflBuffer[index] = INFINITY;
		return;
	}

	real result = INFINITY;
	for (int side = 0; side < DIM; ++side) {
		int2 iNext = i;
		++iNext[side];
		int indexNext = INDEXV(iNext);
		
		real8 eigenvaluesL = eigenvaluesBuffer[side + DIM * index];
		real8 eigenvaluesR = eigenvaluesBuffer[side + DIM * indexNext];
		
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

		real dum = dx[side] / (maxLambda - minLambda);
		result = min(result, dum);
	}
		
	cflBuffer[index] = cfl * result;
}

__kernel void calcDeltaQTilde(
	__global real8* deltaQTildeBuffer,
	const __global real16* eigenvectorsInverseBuffer,
	const __global real8* stateBuffer,
	int2 size,
	real2 dx)
{
	int2 i = (int2)(get_global_id(0), get_global_id(1));
	if (i.x < 2 || i.x >= size.x - 1 
#if DIM > 1
		|| i.y < 2 || i.y >= size.y - 1
#endif
	) return;
	int index = INDEXV(i);

	for (int side = 0; side < DIM; ++side) {
		int2 iPrev = i;
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

__kernel void calcFlux(
	__global real8* fluxBuffer,
	const __global real8* stateBuffer,
	const __global real8* eigenvaluesBuffer,
	const __global real16* eigenvectorsBuffer,
	const __global real16* eigenvectorsInverseBuffer,
	const __global real8* deltaQTildeBuffer,
	int2 size,
	real2 dx,
	__global real *dt)
{
	real2 dt_dx = *dt / dx;
	
	int2 i = (int2)(get_global_id(0), get_global_id(1));
	if (i.x < 2 || i.x >= size.x - 1 
#if 0
		|| i.y < 2 || i.y >= size.y - 1
#endif
	) return;
	int index = INDEXV(i);
	
	for (int side = 0; side < DIM; ++side) {	
		int2 iPrev = i;
		--iPrev[side];
		int indexPrev = INDEXV(iPrev);

		int2 iNext = i;
		++iNext[side];
		int indexNext = INDEXV(iNext);
	
		real8 stateL = stateBuffer[indexPrev];
		real8 stateR = stateBuffer[index];

		int interfaceLIndex = side + DIM * indexPrev;
		int interfaceIndex = side + DIM * index;
		int interfaceRIndex = side + DIM * indexNext;

		real8 deltaQTildeL = deltaQTildeBuffer[interfaceLIndex];
		real8 deltaQTilde = deltaQTildeBuffer[interfaceIndex];
		real8 deltaQTildeR = deltaQTildeBuffer[interfaceRIndex];

		real8 eigenvalues = eigenvaluesBuffer[interfaceIndex];

		real8 theta = step(0.f, eigenvalues) * 2.f - 1.f;
		real8 rTilde = mix(deltaQTildeR, deltaQTildeL, theta * .5f + .5f) / deltaQTilde;
		real8 qAvg = (stateR + stateL) * .5f;
		real8 fluxAvgTilde;
		fluxAvgTilde = matmul(
			eigenvectorsInverseBuffer[0 + 4 * interfaceIndex], 
			eigenvectorsInverseBuffer[1 + 4 * interfaceIndex], 
			eigenvectorsInverseBuffer[2 + 4 * interfaceIndex], 
			eigenvectorsInverseBuffer[3 + 4 * interfaceIndex], 
			qAvg) * eigenvalues;
		real8 phi = slopeLimiter(rTilde);
		real8 epsilon = eigenvalues * dt_dx[side];
		real8 deltaFluxTilde = eigenvalues * deltaQTilde;
		real8 fluxTilde = fluxAvgTilde - .5f * deltaFluxTilde * (theta + .5f * phi * (epsilon - theta));
		
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
	int2 size,
	real2 dx,
	const __global real* dt)
{
	real2 dt_dx = *dt / dx;
	
	int2 i = (int2)(get_global_id(0), get_global_id(1));
	if (i.x < 2 || i.x >= size.x - 2 
#if 0
		|| i.y < 2 || i.y >= size.y - 2
#endif
	) return;
	int index = INDEXV(i);
	
	for (int side = 0; side < DIM; ++side) {	
		int2 iNext = i;
		++iNext[side];
		int indexNext = INDEXV(iNext);
		
		real8 fluxL = fluxBuffer[side + DIM * index];
		real8 fluxR = fluxBuffer[side + DIM * indexNext];

		real8 df = fluxR - fluxL;
		stateBuffer[index] -= df * dt_dx[side];
	}
}

