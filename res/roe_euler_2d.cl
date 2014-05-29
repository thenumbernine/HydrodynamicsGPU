#include "roe_euler_2d.h"

real4 matmul(real16 m, real4 v) {
	return (real4)(
		dot(m.s0123, v),
		dot(m.s4567, v),
		dot(m.s89AB, v),
		dot(m.sCDEF, v));
}

__kernel void calcEigenDecomposition(
	__global Cell* cells,
	int2 size)
{
	int2 i = (int2)(get_global_id(0), get_global_id(1));
	if (i.x >= size.x || i.y >= size.y) return;
	
	int index = i.x + size.x * i.y;
	__global Cell *cell = cells + index;
	
	for (int side = 0; side < 2; ++side) {	
		__global Interface *interface = cell->interfaces + side;
		if (interface->solid) continue;
		
		int2 iPrev = i;
		iPrev[side] = (iPrev[side] + size[side] - 1) % size[side];
		int indexPrev = iPrev.x + size.x * iPrev.y;
		
		__global Cell *cellL = cells + indexPrev;
		__global Cell *cellR = cell;
		
		real2 normal = (real2)(0.f, 0.f);
		normal[side] = 1;

		real densityL = cellL->q[0];
		real invDensityL = 1.f / densityL;
		real2 velocityL = (real2)(cellL->q[1], cellL->q[2]) * invDensityL;
		real energyTotalL = cellL->q[3] * invDensityL;

		real densityR = cellR->q[0];
		real invDensityR = 1.f / densityR;
		real2 velocityR = (real2)(cellR->q[1], cellR->q[2]) * invDensityR;
		real energyTotalR = cellR->q[3] * invDensityR;

		real energyKineticL = .5f * dot(velocityL, velocityL);
		real energyThermalL = energyTotalL - energyKineticL;
		real pressureL = (GAMMA - 1.f) * densityL * energyThermalL;
		real enthalpyTotalL = energyTotalL + pressureL * invDensityL;
		real weightL = sqrt(densityL);

		real energyKineticR = .5f * dot(velocityR, velocityR);
		real energyThermalR = energyTotalR - energyKineticR;
		real pressureR = (GAMMA - 1.f) * densityR * energyThermalR;
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

		interface->eigenvalues = (real4)(
			velocityN - speedOfSound,
			velocityN,
			velocityN,
			velocityN + speedOfSound);

		//min col 
		interface->eigenvectors.s048C = (real4)(
			1.f,
			velocity.x - speedOfSound * normal.x,
			velocity.y - speedOfSound * normal.y,
			enthalpyTotal - speedOfSound * velocityN);
		//mid col (normal)
		interface->eigenvectors.s159D = (real4)(
			1.f,
			velocity.x,
			velocity.y,
			.5f * velocitySq);
		//mid col (tangent)
		interface->eigenvectors.s26AE = (real4)(
			0.f,
			tangent.x,
			tangent.y,
			velocityT);
		//max col 
		interface->eigenvectors.s37BF = (real4)(
			1.f,
			velocity.x + speedOfSound * normal.x,
			velocity.y + speedOfSound * normal.y,
			enthalpyTotal + speedOfSound * velocityN);
		
		//calculate eigenvector inverses ... 
		real invDenom = .5f / (speedOfSound * speedOfSound);
		//min row
		interface->eigenvectorsInverse.s0123 = (real4)(
			(.5f * (GAMMA - 1.f) * velocitySq + speedOfSound * velocityN) * invDenom,
			-(normal.x * speedOfSound + (GAMMA - 1.f) * velocity.x) * invDenom,
			-(normal.y * speedOfSound + (GAMMA - 1.f) * velocity.y) * invDenom,
			(GAMMA - 1.f) * invDenom);
		//mid normal row
		interface->eigenvectorsInverse.s4567 = (real4)(
			1.f - (GAMMA - 1.f) * velocitySq * invDenom,
			(GAMMA - 1.f) * velocity.x * 2.f * invDenom,
			(GAMMA - 1.f) * velocity.y * 2.f * invDenom,
			-(GAMMA - 1.f) * 2.f * invDenom);
		//mid tangent row
		interface->eigenvectorsInverse.s89AB = (real4)(
			-velocityT, 
			tangent.x,
			tangent.y,
			0.f);
		//max row
		interface->eigenvectorsInverse.sCDEF = (real4)(
			(.5f * (GAMMA - 1.f) * velocitySq - speedOfSound * velocityN) * invDenom,
			(normal.x * speedOfSound - (GAMMA - 1.f) * velocity.x) * invDenom,
			(normal.y * speedOfSound - (GAMMA - 1.f) * velocity.y) * invDenom,
			(GAMMA - 1.f) * invDenom);
	}
}

__kernel void calcCFLAndDeltaQTilde(
	__global Cell *cells,
	int2 size,
	__global real *cfl,
	real2 dx)
{
	int2 i = (int2)(get_global_id(0), get_global_id(1));
	if (i.x >= size.x || i.y >= size.y) return;

	int index = i.x + size.x * i.y;
	__global Cell *cell = cells + index;

	for (int side = 0; side < 2; ++side) {
		int2 iNext = i;
		iNext[side] = (iNext[side] + 1) % size[side];
		int indexR = iNext.x + size.x * iNext.y;

		int2 iPrev = i;
		iPrev[side] = (iPrev[side] + size[side] - 1) % size[side];
		int indexPrev = iPrev.x + size.x * iPrev.y;
			
		{
			__global Interface *interfaceL = cell->interfaces + side;
			__global Interface *interfaceR = &cells[indexR].interfaces[side];
		
			real maxLambda = max(
				max(
					interfaceL->eigenvalues.x,
					interfaceL->eigenvalues.y), 
				max(
					interfaceL->eigenvalues.z,
					interfaceL->eigenvalues.w));

			real minLambda = min(
				min(
					interfaceR->eigenvalues.x,
					interfaceR->eigenvalues.y),
				min(
					interfaceR->eigenvalues.z,
					interfaceR->eigenvalues.w));
		
			cfl[index] = dx[side] / (maxLambda - minLambda);
		}

		{
			__global Interface *interface = cell->interfaces + side;
			__global Cell *cellL = cells + indexPrev;
			__global Cell *cellR = cell;

			real4 deltaQ = cellR->q - cellL->q;
			interface->deltaQTilde = matmul(interface->eigenvectorsInverse, deltaQ);
		}
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

__kernel void calcRTilde(
	__global Cell* cells,
	int2 size) 
{
	int2 i = (int2)(get_global_id(0), get_global_id(1));
	if (i.x >= size.x || i.y >= size.y) return;
	
	int index = i.x + size.x * i.y;
	__global Cell *cell = cells + index;
	
	for (int side = 0; side < 2; ++side) {	
		
		int2 iPrev = i;
		iPrev[side] = (iPrev[side] + size[side] - 1) % size[side];
		int indexPrev = iPrev.x + size.x * iPrev.y;
		
		int2 iNext = i;
		iNext[side] = (iNext[side] + 1) % size[side];
		int indexNext = iNext.x + size.x * iNext.y;
		
		__global Interface *interfaceL = &cells[indexPrev].interfaces[side];
		__global Interface *interface = &cells[index].interfaces[side];
		__global Interface *interfaceR = &cells[indexNext].interfaces[side];

		for (int state = 0; state < 4; ++state) {
			if (fabs(interface->deltaQTilde[state]) > 0.f) {
				if (interface->eigenvalues[state] > 0.f) {
					interface->rTilde[state] = interfaceL->deltaQTilde[state] / interface->deltaQTilde[state];
				} else {
					interface->rTilde[state] = interfaceR->deltaQTilde[state] / interface->deltaQTilde[state];
				}
			} else {
				interface->rTilde[state] = 0.f;
			}
		}
	}
}

real4 fluxMethod(real4 r) {
	//superbee
	return max(0.f, max(min(1.f, 2.f * r), min(2.f, r)));
}

__kernel void calcFlux(
	__global Cell* cells,
	int2 size,
	real2 dx,
	__global real *dt)
{
	real2 dt_dx = *dt / dx;
	
	int2 i = (int2)(get_global_id(0), get_global_id(1));
	if (i.x >= size.x || i.y >= size.y) return;

	int index = i.x + size.x * i.y;
	__global Cell *cell = cells + index;
	
	for (int side = 0; side < 2; ++side) {	
		int2 iPrev = i;
		iPrev[side] = (iPrev[side] + size[side] - 1) % size[side];
		int indexPrev = iPrev.x + size.x * iPrev.y;
		
		__global Cell *cellL = cells + indexPrev;
		__global Cell *cellR = cell;
		__global Interface *interface = cell->interfaces + side;

		real4 qAvg = (cellR->q + cellL->q) * .5f;
	
		real4 fluxAvgTilde = matmul(interface->eigenvectorsInverse, qAvg);
		fluxAvgTilde *= interface->eigenvalues;

		real4 theta = step(0.f, interface->eigenvalues) * 2.f - 1.f;
		real4 phi = fluxMethod(interface->rTilde);
		real4 epsilon = interface->eigenvalues * dt_dx[side];
		real4 deltaFluxTilde = interface->eigenvalues * interface->deltaQTilde;
		real4 fluxTilde = fluxAvgTilde - .5f * deltaFluxTilde * (theta + phi * (epsilon - theta));
		
		interface->flux = matmul(interface->eigenvectors, fluxTilde);
	}
}

__kernel void updateState(
	__global Cell* cells,
	int2 size,
	real2 dx,
	__global real* dt)
{
	real2 dt_dx = *dt / dx;
	
	int2 i = (int2)(get_global_id(0), get_global_id(1));
	if (i.x >= size.x || i.y >= size.y) return;

	int index = i.x + size.x * i.y;
	__global Cell *cell = cells + index;
	
	for (int side = 0; side < 2; ++side) {	
		int2 iNext = i;
		iNext[side] = (iNext[side] + 1) % size[side];
		int indexNext = iNext.x + size.x * iNext.y;
		
		__global Interface *interfaceL = &cells[index].interfaces[side];
		__global Interface *interfaceR = &cells[indexNext].interfaces[side];

		real4 df = interfaceR->flux - interfaceL->flux;
		cell->q -= df * dt_dx[side];
	}
}

__kernel void convertToTex(
	__global Cell* cells,
	int2 size,
	__write_only image2d_t fluidTex,
	__read_only image2d_t gradientTex)
{
	int2 i = (int2)(get_global_id(0), get_global_id(1));
	if (i.x >= size.x || i.y >= size.y) return;
	
	int index = i.x + size.x * i.y;
	__global Cell *cell = cells + index;

	float4 color = read_imagef(gradientTex, 
		CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_NONE | CLK_FILTER_LINEAR,
		(float2)(cell->q[0] * 2.f, .5f));
	write_imagef(fluidTex, i, color.bgra);
}

