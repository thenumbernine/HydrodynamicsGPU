#include "HydroGPU/Shared/Types.h"

real4 matmul(real16 m, real4 v);
real4 fluxMethod(real4 r);

real4 matmul(real16 m, real4 v) {
	return (real4)(
		dot(m.s0123, v),
		dot(m.s4567, v),
		dot(m.s89AB, v),
		dot(m.sCDEF, v));
}

__kernel void calcCFL(
	__global real* cflBuffer,
	const __global real4* stateBuffer,
	int2 size,
	real2 dx)
{
	int2 i = (int2)(get_global_id(0), get_global_id(1));
	if (i.x >= size.x || i.y >= size.y) return;
	int index = i.x + size.x * i.y;

	real4 state = stateBuffer[index];
	real density = state.x;
	real2 velocity = state.yz / density;
	real specificTotalEnergy = state.w / density;
	real specificKineticEnergy = .5 * dot(velocity, velocity);
	real specificInternalEnergy = specificTotalEnergy - specificKineticEnergy;
	real speedOfSound = sqrt(GAMMA * (GAMMA - 1.) * specificInternalEnergy);
	dum.x = dx.x / (speedOfSound + Math.abs(velocity.x));
	dum.y = dx.y / (speedOfSound + Math.abs(velocity.y));
	cflBuffer[index] = min(dum.x, dum.y);
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

__kernel void calcInterfaceVelocity(
	__global real4* interfaceVelocityBuffer,
	const __global real4* stateBuffer,
	int2 size,
	real2 dx)
{
	int2 i = (int2)(get_global_id(0), get_global_id(1));
	if (i.x >= size.x || i.y >= size.y) return;
	int index = i.x + size.x * i.y;

	real2 interfaceVelocity;
	for (int side = 0; side < 2; ++side) {
		int2 iPrev = i;
		iPrev[side] = (iPrev[side] + size[side] - 1) % size[side];
		int indexPrev = iPrev.x + size.x * iPrev.y;
				
		real4 stateL = stateBuffer[indexPrev];
		real densityL = stateL.x;
		real velocityL = stateL.yz[side] / densityL;

		real4 stateR = stateBuffer[index];
		real densityR = stateR.x;
		real velocityR = stateR.yz[side] / densityR;

		interfaceVelocity[side] = .5 * (velocityL + velocityR);
	}
	interfaceVelocityBuffer[index] = interfaceVelocity;
}

real4 fluxMethod(real4 r) {
	//superbee
	return max(0.f, max(min(1.f, 2.f * r), min(2.f, r)));
}

__kernel void calcFlux(
	__global real4* fluxBuffer,
	const __global real4* stateBuffer,
	const __global real4* interfaceVelocityBuffer,
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

		real4 egz = step(0.f, eigenvalues);
		real4 rTilde = mix(deltaQTildeR, deltaQTildeL, egz) / deltaQTilde;
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
	__global real* dt)
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

	float4 color = read_imagef(gradientTex, 
		CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_NONE | CLK_FILTER_LINEAR,
		(float2)(state.w * 2.f, .5f));
	write_imagef(fluidTex, i, color.bgra);
}

__kernel void addDrop(
	__global real4* stateBuffer,
	int2 size,
	real2 xmin,
	real2 xmax,
	__global real* dt,
	real2 pos,
	real2 sourceVelocity)
{
	int2 i = (int2)(get_global_id(0), get_global_id(1));
	if (i.x >= size.x || i.y >= size.y) return;

	int index = i.x + size.x * i.y;
	real4 state = stateBuffer[index];

	float dropRadius = .02f;
	float densityMagnitude = .05f;
	float velocityMagnitude = 1.f;
	float energyThermalMagnitude = 0.f;

	real2 x = (real2)(i.x, i.y) / (real2)(size.x, size.y) * (xmax - xmin) + xmin;
	real2 dx = (x - pos) / dropRadius;
	float rSq = dot(dx, dx);
	float falloff = exp(-rSq);

	real density = state.x;
	real2 velocity = state.yz / density;
	real energyTotal = state.w / density;
	real energyKinetic = .5f * dot(velocity, velocity);
	real energyThermal = energyTotal - energyKinetic;

	density += densityMagnitude * falloff;
	velocity += sourceVelocity * (falloff * velocityMagnitude);
	energyThermal += energyThermalMagnitude * falloff;

	energyKinetic = .5f * dot(velocity, velocity);
	energyTotal = energyThermal + energyKinetic;

	stateBuffer[index] = (real4)(1.f, velocity.x, velocity.y, energyTotal) * density;
}

__kernel void addSource(
	__global real4* stateBuffer,
	int2 size,
	real2 xmin,
	real2 xmax,
	__global real* dt)
{
return;// working on this
#if 0
	int2 i = (int2)(get_global_id(0), get_global_id(1));
	if (i.x >= size.x || i.y >= size.y) return;

	int index = i.x + size.x * i.y;
	__global Cell *cell = cells + index;

	real2 x = (real2)i / (real2)size * (xmax - xmin) + xmin;
	real dx2 = dot(x,x);
	real infl = exp(-10000.f * dx2);

	cell->q[1] += infl * *dt; 
#endif
}
