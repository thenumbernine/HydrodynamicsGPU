#include "HydroGPU/Shared/Types.h"

real4 matmul(real16 m, real4 v);
real mat44det(real16 m);
real16 mat44inv(real16 m);
real4 fluxMethod(real4 r);

real4 matmul(real16 m, real4 v) {
	return (real4)(
		dot(m.s0123, v),
		dot(m.s4567, v),
		dot(m.s89AB, v),
		dot(m.sCDEF, v));
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
	__global real4* eigenvaluesBuffer,
	__global real16* eigenvectorsBuffer,
	__global real16* eigenvectorsInverseBuffer,
	const __global real4* stateBuffer,
	int2 size)
{
	int2 i = (int2)(get_global_id(0), get_global_id(1));
	if (i.x >= size.x || i.y >= size.y) return;
	int index = i.x + size.x * i.y;

	for (int side = 0; side < 2; ++side) {	
		int2 iPrev = i;
		iPrev[side] = (iPrev[side] + size[side] - 1) % size[side];
		int indexPrev = iPrev.x + size.x * iPrev.y;

		int interfaceIndex = side + 2 * index;

		real4 stateL = stateBuffer[indexPrev];
		real4 stateR = stateBuffer[index];
	
#if 0	//calculate flux based on normal

		real2 normal = (real2)(0.f, 0.f);
		normal[side] = 1;

		real densityL = stateL.x;
		real invDensityL = 1.f / densityL;
		real2 velocityL = stateL.yz * invDensityL;
		real energyTotalL = stateL.w * invDensityL;

		real densityR = stateR.x;
		real invDensityR = 1.f / densityR;
		real2 velocityR = stateR.yz * invDensityR;
		real energyTotalR = stateR.w * invDensityR;

		real energyKineticL = .5f * dot(velocityL, velocityL);
		real energyInternalL = energyTotalL - energyKineticL;
		real pressureL = (GAMMA - 1.f) * densityL * energyInternalL;
		real enthalpyTotalL = energyTotalL + pressureL * invDensityL;
		real weightL = sqrt(densityL);

		real energyKineticR = .5f * dot(velocityR, velocityR);
		real energyInternalR = energyTotalR - energyKineticR;
		real pressureR = (GAMMA - 1.f) * densityR * energyInternalR;
		real enthalpyTotalR = energyTotalR + pressureR * invDensityR;
		real weightR = sqrt(densityR);

		real roeWeightNormalization = 1.f / (weightL + weightR);
		real2 velocity = (weightL * velocityL + weightR * velocityR) * roeWeightNormalization;
		real enthalpyTotal = (weightL * enthalpyTotalL + weightR * enthalpyTotalR) * roeWeightNormalization;
		
		real velocitySq = dot(velocity, velocity);
		real speedOfSound = sqrt((GAMMA - 1.f) * (enthalpyTotal - .5f * velocitySq));
		
		real2 tangent = (real2)(-normal.y, normal.x);
		real velocityN = dot(velocity, normal);
		real velocityT = dot(velocity, tangent);
	
		//eigenvalues

		eigenvaluesBuffer[interfaceIndex] =  (real4)(
			velocityN - speedOfSound,
			velocityN,
			velocityN,
			velocityN + speedOfSound);

		//eigenvectors

		//specify transposed
		real16 eigenvectors = (real16)(
		//min col 
			1.f,
			velocity.x - speedOfSound * normal.x,
			velocity.y - speedOfSound * normal.y,
			enthalpyTotal - speedOfSound * velocityN,
		//mid col (normal)
			1.f,
			velocity.x,
			velocity.y,
			.5f * velocitySq,
		//mid col (tangent)
			0.f,
			tangent.x,
			tangent.y,
			velocityT,
		//max col 
			1.f,
			velocity.x + speedOfSound * normal.x,
			velocity.y + speedOfSound * normal.y,
			enthalpyTotal + speedOfSound * velocityN);

		//transpose and store
		eigenvectorsBuffer[interfaceIndex] = (real16)(
			eigenvectors.s048C,
			eigenvectors.s159D,
			eigenvectors.s26AE,
			eigenvectors.s37BF);

#if 0	//analytical eigenvalues.  getting worse results on double precision on my CPU-driven hydrodynamics program
		//calculate eigenvector inverses ... 
		real invDenom = .5f / (speedOfSound * speedOfSound);
		eigenvectorsInverseBuffer[interfaceIndex] = (real16)( 
		//min row
			(.5f * (GAMMA - 1.f) * velocitySq + speedOfSound * velocityN) * invDenom,
			-(normal.x * speedOfSound + (GAMMA - 1.f) * velocity.x) * invDenom,
			-(normal.y * speedOfSound + (GAMMA - 1.f) * velocity.y) * invDenom,
			(GAMMA - 1.f) * invDenom,
		//mid normal row
			1.f - (GAMMA - 1.f) * velocitySq * invDenom,
			(GAMMA - 1.f) * velocity.x * 2.f * invDenom,
			(GAMMA - 1.f) * velocity.y * 2.f * invDenom,
			-(GAMMA - 1.f) * 2.f * invDenom,
		//mid tangent row
			-velocityT, 
			tangent.x,
			tangent.y,
			0.f,
		//max row
			(.5f * (GAMMA - 1.f) * velocitySq - speedOfSound * velocityN) * invDenom,
			(normal.x * speedOfSound - (GAMMA - 1.f) * velocity.x) * invDenom,
			(normal.y * speedOfSound - (GAMMA - 1.f) * velocity.y) * invDenom,
			(GAMMA - 1.f) * invDenom);
#endif
#if 1 //numerically solve for the inverse
		eigenvectorsInverseBuffer[interfaceIndex] = mat44inv(eigenvectorsBuffer[interfaceIndex]);
#endif
#endif





#if 1	//calculate flux in x-axis and rotate into normal

		real densityL = stateL.x;
		real invDensityL = 1.f / densityL;
		real2 velocityL = stateL.yz * invDensityL;
		real energyTotalL = stateL.w * invDensityL;

		real densityR = stateR.x;
		real invDensityR = 1.f / densityR;
		real2 velocityR = stateR.yz * invDensityR;
		real energyTotalR = stateR.w * invDensityR;

		real energyKineticL = .5f * dot(velocityL, velocityL);
		real energyInternalL = energyTotalL - energyKineticL;
		real pressureL = (GAMMA - 1.f) * densityL * energyInternalL;
		real enthalpyTotalL = energyTotalL + pressureL * invDensityL;
		real weightL = sqrt(densityL);

		real energyKineticR = .5f * dot(velocityR, velocityR);
		real energyInternalR = energyTotalR - energyKineticR;
		real pressureR = (GAMMA - 1.f) * densityR * energyInternalR;
		real enthalpyTotalR = energyTotalR + pressureR * invDensityR;
		real weightR = sqrt(densityR);

		real roeWeightNormalization = 1.f / (weightL + weightR);
		real2 velocity = (weightL * velocityL + weightR * velocityR) * roeWeightNormalization;
		real enthalpyTotal = (weightL * enthalpyTotalL + weightR * enthalpyTotalR) * roeWeightNormalization;
		
		real velocitySq = dot(velocity, velocity);
		real speedOfSound = sqrt((GAMMA - 1.f) * (enthalpyTotal - .5f * velocitySq));

		if (side == 1) {
			velocity.xy = (real2)(velocity.y, -velocity.x);	// -90' rotation to put the y axis contents into the x axis
		}

		//eigenvalues

		eigenvaluesBuffer[interfaceIndex] =  (real4)(
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
		
		eigenvectorsBuffer[interfaceIndex] = eigenvectors;
		eigenvectorsInverseBuffer[interfaceIndex] = eigenvectorsInverse;
#endif
	}
}

__kernel void calcCFL(
	__global real* cflBuffer,
	const __global real4* eigenvaluesBuffer,
	int2 size,
	real2 dx)
{
	int2 i = (int2)(get_global_id(0), get_global_id(1));
	if (i.x >= size.x || i.y >= size.y) return;
	int index = i.x + size.x * i.y;

	real2 dum;
	for (int side = 0; side < 2; ++side) {
		int2 iNext = i;
		iNext[side] = (iNext[side] + 1) % size[side];
		int indexNext = iNext.x + size.x * iNext.y;
		
		real4 eigenvaluesL = eigenvaluesBuffer[side + 2 * index];
		real4 eigenvaluesR = eigenvaluesBuffer[side + 2 * indexNext];
		
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

		dum[side] = dx[side] / (maxLambda - minLambda);
	}
		
	cflBuffer[index] = min(dum.x, dum.y);
}

__kernel void calcDeltaQTilde(
	__global real4* deltaQTildeBuffer,
	const __global real16* eigenvectorsInverseBuffer,
	const __global real4* stateBuffer,
	int2 size,
	real2 dx)
{
	int2 i = (int2)(get_global_id(0), get_global_id(1));
	if (i.x >= size.x || i.y >= size.y) return;
	int index = i.x + size.x * i.y;

	for (int side = 0; side < 2; ++side) {
		int2 iPrev = i;
		iPrev[side] = (iPrev[side] + size[side] - 1) % size[side];
		int indexPrev = iPrev.x + size.x * iPrev.y;
				
		real4 stateL = stateBuffer[indexPrev];
		real4 stateR = stateBuffer[index];

		int interfaceIndex = side + 2 * index;
		
		real4 deltaQ = stateR - stateL;
		deltaQTildeBuffer[interfaceIndex] = matmul(eigenvectorsInverseBuffer[interfaceIndex], deltaQ);
	}
}


__kernel void calcCFLMinReduce(
	__global real *cflDst, 
	__local real *cflSrc) 
{
	int lid = get_local_id(0);
	int group_size = get_local_size(0);

	cflSrc[lid] = cflDst[get_global_id(0)];
	barrier(CLK_LOCAL_MEM_FENCE);

	for(int i = group_size/2; i>0; i >>= 1) {
		if(lid < i) {
			 cflSrc[lid] = min(cflSrc[lid], cflSrc[lid + i]);
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if(lid == 0) {
		cflDst[get_group_id(0)] = cflSrc[0];
	}
}

__kernel void calcCFLMinFinal(
	__global real *cflDst, 
	__local real *cflSrc, 
	__global real *result,
	real cfl,
	size_t group_size)
{
	int lid = get_local_id(0);

	cflSrc[lid] = cflDst[get_local_id(0)];
	barrier(CLK_LOCAL_MEM_FENCE);

	for(int i = group_size/2; i>0; i >>= 1) {
		if(lid < i) {
			cflSrc[lid] = min(cflSrc[lid], cflSrc[lid + i]);
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if(lid == 0) {
		*result = cflSrc[0] * cfl;
	}
}

real4 fluxMethod(real4 r) {
	//superbee
	return max(0.f, max(min(1.f, 2.f * r), min(2.f, r)));
}

__kernel void calcFlux(
	__global real4* fluxBuffer,
	const __global real4* stateBuffer,
	const __global real4* eigenvaluesBuffer,
	const __global real16* eigenvectorsBuffer,
	const __global real16* eigenvectorsInverseBuffer,
	const __global real4* deltaQTildeBuffer,
	int2 size,
	real2 dx,
	__global real *dt)
{
	real2 dt_dx = *dt / dx;
	
	int2 i = (int2)(get_global_id(0), get_global_id(1));
	if (i.x >= size.x || i.y >= size.y) return;

	int index = i.x + size.x * i.y;
	
	for (int side = 0; side < 2; ++side) {	
		int2 iPrev = i;
		iPrev[side] = (iPrev[side] + size[side] - 1) % size[side];
		int indexPrev = iPrev.x + size.x * iPrev.y;

		int2 iNext = i;
		iNext[side] = (iNext[side] + 1) % size[side];
		int indexNext = iNext.x + size.x * iNext.y;
	
		real4 stateL = stateBuffer[indexPrev];
		real4 stateR = stateBuffer[index];

		int interfaceLIndex = side + 2 * indexPrev;
		int interfaceIndex = side + 2 * index;
		int interfaceRIndex = side + 2 * indexNext;

		real4 deltaQTildeL = deltaQTildeBuffer[interfaceLIndex];
		real4 deltaQTilde = deltaQTildeBuffer[interfaceIndex];
		real4 deltaQTildeR = deltaQTildeBuffer[interfaceRIndex];

		real4 eigenvalues = eigenvaluesBuffer[interfaceIndex];
		real16 eigenvectors = eigenvectorsBuffer[interfaceIndex];
		real16 eigenvectorsInverse = eigenvectorsInverseBuffer[interfaceIndex];

		real4 eigenvaluesGreaterThanZero = step(0.f, eigenvalues);
		real4 rTilde = mix(deltaQTildeR, deltaQTildeL, eigenvaluesGreaterThanZero) / deltaQTilde;
		real4 qAvg = (stateR + stateL) * .5f;
		real4 fluxAvgTilde = matmul(eigenvectorsInverse, qAvg) * eigenvalues;
		real4 theta = step(0.f, eigenvalues) * 2.f - 1.f;
		real4 phi = fluxMethod(rTilde);
		real4 epsilon = eigenvalues * dt_dx[side];
		real4 deltaFluxTilde = eigenvalues * deltaQTilde;
		real4 fluxTilde = fluxAvgTilde - .5f * deltaFluxTilde * (theta + phi * (epsilon - theta));
		
		fluxBuffer[side + 2 * index] = matmul(eigenvectors, fluxTilde);
	}
}

__kernel void updateState(
	__global real4* stateBuffer,
	const __global real4* fluxBuffer,
	int2 size,
	real2 dx,
	const __global real* dt)
{
	real2 dt_dx = *dt / dx;
	
	int2 i = (int2)(get_global_id(0), get_global_id(1));
	if (i.x >= size.x || i.y >= size.y) return;

	int index = i.x + size.x * i.y;
	
	for (int side = 0; side < 2; ++side) {	
		int2 iNext = i;
		iNext[side] = (iNext[side] + 1) % size[side];
		int indexNext = iNext.x + size.x * iNext.y;
		
		real4 fluxL = fluxBuffer[side + 2 * index];
		real4 fluxR = fluxBuffer[side + 2 * indexNext];

		real4 df = fluxR - fluxL;
		stateBuffer[index] -= df * dt_dx[side];
	}
}

__kernel void convertToTex(
	__global real4* stateBuffer,
	int2 size,
	__write_only image2d_t fluidTex,
	__read_only image2d_t gradientTex)
{
	int2 i = (int2)(get_global_id(0), get_global_id(1));
	if (i.x >= size.x || i.y >= size.y) return;
	
	int index = i.x + size.x * i.y;
	real4 state = stateBuffer[index];
	
	//density coloring
	real value = state.x;

	//velocity coloring
	//real2 velocity = state.yz / state.x;
	//real value = log(dot(velocity, velocity) + 1.f);

	float4 color = read_imagef(gradientTex, 
		CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_NONE | CLK_FILTER_LINEAR,
		(float2)(value * 2.f, .5f));
	write_imagef(fluidTex, i, color.bgra);
}

__kernel void addDrop(
	__global real4* stateBuffer,
	int2 size,
	real2 xmin,
	real2 xmax,
	const __global real* dt,
	real2 pos,
	real2 sourceVelocity)
{
	int2 i = (int2)(get_global_id(0), get_global_id(1));
	if (i.x >= size.x || i.y >= size.y) return;

	int index = i.x + size.x * i.y;
	real4 state = stateBuffer[index];

	const float dropRadius = .02f;
	const float densityMagnitude = 50.f;
	const float velocityMagnitude = 10000.f;
	const float energyInternalMagnitude = 0.f;
	
	real cellPosX = (real)i.x / (real)size.x * (xmax.x - xmin.x) + xmin.x;
	real cellPosY = (real)i.y / (real)size.y * (xmax.y - xmin.y) + xmin.y;
	real2 cellPos = (real2)(cellPosX, cellPosY);
	real2 dx = (cellPos - pos) / dropRadius;
	float rSq = dot(dx, dx);
	float falloff = exp(-rSq);
	
	real density = state.x;
	real2 velocity = state.yz / density;
	real energyTotal = state.w / density;
	real energyKinetic = .5f * dot(velocity, velocity);
	real energyInternal = energyTotal - energyKinetic;

	density += *dt * densityMagnitude * falloff;
	velocity += *dt * sourceVelocity * (falloff * velocityMagnitude);
	energyInternal += *dt * energyInternalMagnitude * falloff;

	energyKinetic = .5f * dot(velocity, velocity);
	energyTotal = energyInternal + energyKinetic;

	stateBuffer[index] = (real4)(1.f, velocity.x, velocity.y, energyTotal) * density;
}

__kernel void addSource(
	__global real4* stateBuffer,
	int2 size,
	real2 xmin,
	real2 xmax,
	const __global real* dt)
{
	int2 i = (int2)(get_global_id(0), get_global_id(1));
	if (i.x >= size.x || i.y >= size.y) return;

	const float radius = .05f;

	real cellPosX = (real)i.x / (real)size.x * (xmax.x - xmin.x) + xmin.x;
	real cellPosY = (real)i.y / (real)size.y * (xmax.y - xmin.y) + xmin.y;
	real2 cellPos = (real2)(cellPosX, cellPosY);
	real2 sourcePos = (real2)(xmin.x, .5f * (xmax.y + xmin.y));
	real2 dx = (cellPos - sourcePos) / radius;
	real rSq = dot(dx,dx);
	real falloff = exp(-rSq);

	int index = i.x + size.x * i.y;
	real4 state = stateBuffer[index];
	real density = state.x;
	real2 velocity = state.yz / density;
	real energyTotal = state.w / density;
	real energyKinetic = .5f * dot(velocity, velocity);
	real energyInternal = energyTotal - energyKinetic;

	const float velocityMagnitude = 10.f;
	velocity.x += velocityMagnitude * falloff * *dt;

	energyKinetic = .5f * dot(velocity, velocity);
	energyTotal = energyInternal + energyKinetic;

	stateBuffer[index] = (real4)(1.f, velocity.x, velocity.y, energyTotal) * density;
}

