#include "HydroGPU/Shared/Common.h"

//prototypes

void calcInterfaceVelocityAt(
	__global real2* interfaceVelocityBuffer,	//uses ghost
	const __global real4* stateBuffer,	//uses ghost
	int2 size,
	real2 dx,
	int2 i);

real4 slopeLimiter(real4 r);

void calcFluxAt(
	__global real4* fluxBuffer,	//ghost
	const __global real4* stateBuffer,	//ghost
	const __global real2* interfaceVelocityBuffer,	//ghost
	int2 size,
	real2 dt_dx,
	int2 i);

void computePressureAt(
	__global real* pressureBuffer,	//ghost
	const __global real4* stateBuffer,	//ghost
	const __global real* gravityPotentialBuffer,	//ghost
	int2 size,
	int2 i);

//calc cfl

__kernel void calcCFL(
	__global real* cflBuffer,
	const __global real4* stateBuffer,	//uses ghost
	const __global real* gravityPotentialBuffer,	//uses ghost
	int2 size,
	real2 dx,
	real cfl)
{
	int2 i = (int2)(get_global_id(0), get_global_id(1));
	int cflIndex = INDEXV(i);
	int index = INDEXGHOSTV(i);
	
	real4 state = stateBuffer[index];
	real density = state.x;
	real2 velocity = state.yz / density;
	real specificTotalEnergy = state.w / density;
	real specificKineticEnergy = .5f * dot(velocity, velocity);
	real specificPotentialEnergy = gravityPotentialBuffer[index]; 
	real specificInternalEnergy = specificTotalEnergy - specificKineticEnergy - specificPotentialEnergy;
	real speedOfSound = sqrt(GAMMA * (GAMMA - 1.f) * specificInternalEnergy);
	real dumx = dx.x / (speedOfSound + fabs(velocity.x));
	real dumy = dx.y / (speedOfSound + fabs(velocity.y));
	cflBuffer[cflIndex] = cfl * min(dumx, dumy);
}

//reduce cfl to find minimum

//http://developer.amd.com/resources/documentation-articles/articles-whitepapers/opencl-optimization-case-study-simple-reductions/
__kernel void calcCFLMinReduce(
	const __global real* buffer,
	__local real* scratch,
	__const int length,
	__global real* result)
{
	int global_index = get_global_id(0);
	real accumulator = INFINITY;
	// Loop sequentially over chunks of input vector
	while (global_index < length) {
		real element = buffer[global_index];
		accumulator = (accumulator < element) ? accumulator : element;
		global_index += get_global_size(0);
	}

	// Perform parallel reduction
	int local_index = get_local_id(0);
	scratch[local_index] = accumulator;
	barrier(CLK_LOCAL_MEM_FENCE);
	for(int offset = get_local_size(0) / 2; offset > 0; offset = offset / 2) {
		if (local_index < offset) {
			real other = scratch[local_index + offset];
			real mine = scratch[local_index];
			scratch[local_index] = (mine < other) ? mine : other;
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	if (local_index == 0) {
		result[get_group_id(0)] = scratch[0];
	}
}

//compute interface velocity

void calcInterfaceVelocityAt(
	__global real2* interfaceVelocityBuffer,	//uses ghost
	const __global real4* stateBuffer,	//uses ghost
	int2 size,
	real2 dx,
	int2 i)
{
	int index = INDEXGHOSTV(i);
	
	real2 interfaceVelocity;
	for (int side = 0; side < 2; ++side) {
		int2 iPrev = i;
		--iPrev[side];
		int indexPrev = INDEXGHOSTV(iPrev);
		
		real4 stateL = stateBuffer[indexPrev];
		real densityL = stateL.x;
		real velocityL = stateL[side+1] / densityL;
		
		real4 stateR = stateBuffer[index];
		real densityR = stateR.x;
		real velocityR = stateR[side+1] / densityR;
		
		interfaceVelocity[side] = .5f * (velocityL + velocityR);
	}
	interfaceVelocityBuffer[index] = interfaceVelocity;
}

__kernel void calcInterfaceVelocity(
	__global real2* interfaceVelocityBuffer,
	const __global real4* stateBuffer,
	int2 size,
	real2 dx)
{
	int2 i = (int2)(get_global_id(0), get_global_id(1));
	calcInterfaceVelocityAt(interfaceVelocityBuffer, stateBuffer, size, dx, i);
}

__kernel void calcInterfaceVelocityHorizontal(
	__global real2* interfaceVelocityBuffer,
	const __global real4* stateBuffer,
	int2 size,
	real2 dx,
	int y)
{
	int2 i = (int2)(get_global_id(0), y);
	calcInterfaceVelocityAt(interfaceVelocityBuffer, stateBuffer, size, dx, i);
}

__kernel void calcInterfaceVelocityVertical(
	__global real2* interfaceVelocityBuffer,
	const __global real4* stateBuffer,
	int2 size,
	real2 dx,
	int x)
{
	int2 i = (int2)(x, get_global_id(0));
	calcInterfaceVelocityAt(interfaceVelocityBuffer, stateBuffer, size, dx, i);
}

//calc flux

real4 slopeLimiter(real4 r) {
	//donor cell
	//return (real4)(0.f, 0.f, 0.f, 0.f);
	//Lax-Wendroff
	//return (real4)(1.f, 1.f, 1.f, 1.f);
	//superbee
	return max(0.f, max(min(1.f, 2.f * r), min(2.f, r)));
}

void calcFluxAt(
	__global real4* fluxBuffer,	//ghost
	const __global real4* stateBuffer,	//ghost
	const __global real2* interfaceVelocityBuffer,	//ghost
	int2 size,
	real2 dt_dx,
	int2 i)
{
	int index = INDEXGHOSTV(i);
	
	for (int side = 0; side < 2; ++side) {	
		int2 iPrev = i;
		--iPrev[side];
		int indexPrev = INDEXGHOSTV(iPrev);
	
		int2 iPrev2 = iPrev;
		--iPrev2[side];
		int indexPrev2 = INDEXGHOSTV(iPrev2);
		
		int2 iNext = i;
		++iNext[side];
		int indexNext = INDEXGHOSTV(iNext);

		int indexL2 = indexPrev2;
		int indexL1 = indexPrev;
		int indexR1 = index;
		int indexR2 = indexNext;

		real4 stateL2 = stateBuffer[indexL2];
		real4 stateL = stateBuffer[indexL1];
		real4 stateR = stateBuffer[indexR1];
		real4 stateR2 = stateBuffer[indexR2];

		real4 deltaStateL = stateL - stateL2;
		real4 deltaState = stateR - stateL;
		real4 deltaStateR = stateR2 - stateR;

		real interfaceVelocity = interfaceVelocityBuffer[index][side];
		real theta = step(0.f, interfaceVelocity);
		
		real4 stateSlopeRatio = mix(deltaStateR, deltaStateL, theta) / deltaState;
		real4 phi = slopeLimiter(stateSlopeRatio);
		real4 delta = phi * deltaState;

		real4 flux = mix(stateR, stateL, theta) * interfaceVelocity
			+ delta * .5f * (.5f * fabs(interfaceVelocity) * (1.f - fabs(interfaceVelocity * dt_dx[side])));

		fluxBuffer[side + 2 * index] = flux;
	}
}

__kernel void calcFlux(
	__global real4* fluxBuffer,	//ghost
	const __global real4* stateBuffer,	//ghost
	const __global real2* interfaceVelocityBuffer,	//ghost
	int2 size,
	real2 dx,
	const __global real* dtBuffer)
{
	int2 i = (int2)(get_global_id(0), get_global_id(1));
	real2 dt_dx = dtBuffer[0] / dx;
	calcFluxAt(fluxBuffer, stateBuffer, interfaceVelocityBuffer, size, dt_dx, i);
}

__kernel void calcFluxHorizontal(
	__global real4* fluxBuffer,	//ghost
	const __global real4* stateBuffer,	//ghost
	const __global real2* interfaceVelocityBuffer,	//ghost
	int2 size,
	real2 dx,
	const __global real* dtBuffer,
	int y)
{
	int2 i = (int2)(get_global_id(0), y);
	real2 dt_dx = dtBuffer[0] / dx;
	calcFluxAt(fluxBuffer, stateBuffer, interfaceVelocityBuffer, size, dt_dx, i);
}

__kernel void calcFluxVertical(
	__global real4* fluxBuffer,	//ghost
	const __global real4* stateBuffer,	//ghost
	const __global real2* interfaceVelocityBuffer,	//ghost
	int2 size,
	real2 dx,
	const __global real* dtBuffer,
	int x)
{
	int2 i = (int2)(x, get_global_id(0));
	real2 dt_dx = dtBuffer[0] / dx;
	calcFluxAt(fluxBuffer, stateBuffer, interfaceVelocityBuffer, size, dt_dx, i);
}

//integrate flux

__kernel void integrateFlux(
	__global real4* stateBuffer,	//ghost
	const __global real4* fluxBuffer,	//ghost
	int2 size,
	real2 dx,
	const __global real* dtBuffer)
{
	real2 dt_dx = dtBuffer[0] / dx;
	
	int2 i = (int2)(get_global_id(0), get_global_id(1));
	int index = INDEXGHOSTV(i);
	
	for (int side = 0; side < 2; ++side) {
		int2 iNext = i;
		++iNext[side];
		int indexNext = INDEXGHOSTV(iNext);
		
		real4 fluxL = fluxBuffer[side + 2 * index];
		real4 fluxR = fluxBuffer[side + 2 * indexNext];
	
		real4 df = fluxR - fluxL;
		stateBuffer[index] -= df * dt_dx[side];
	}
}

__kernel void poissonRelax(
	__global real* gravityPotentialBuffer,	//ghost
	const __global real4* stateBuffer,	//ghost
	int2 size,
	real2 dx)
{
	int2 i = (int2)(get_global_id(0), get_global_id(1));
	int index = INDEXGHOSTV(i);

	int xp = INDEXGHOST(i.x - 1, i.y);
	int xn = INDEXGHOST(i.x + 1, i.y);
	int yp = INDEXGHOST(i.x, i.y - 1);
	int yn = INDEXGHOST(i.x, i.y + 1);

	real sum = gravityPotentialBuffer[xp] 
				+ gravityPotentialBuffer[xn] 
				+ gravityPotentialBuffer[yp] 
				+ gravityPotentialBuffer[yn];
	
#define M_PI 3.141592653589793115997963468544185161590576171875f
#define GRAVITY_CONSTANT 1.f		//6.67384e-11 m^3 / (kg s^2)
	real scale = M_PI * GRAVITY_CONSTANT * dx.x * dx.y;

	gravityPotentialBuffer[index] = .25f * sum + scale * stateBuffer[index].x;
}

void computePressureAt(
	__global real* pressureBuffer,	//ghost
	const __global real4* stateBuffer,	//ghost
	const __global real* gravityPotentialBuffer,	//ghost
	int2 size,
	int2 i)
{
	int index = INDEXGHOSTV(i);
	
	real4 state = stateBuffer[index];
	
	real density = state.x;
	real2 velocity = state.yz / density;
	real energyTotal = state.w / density;
	
	real energyKinetic = .5f * dot(velocity, velocity);
	real energyPotential = gravityPotentialBuffer[index];
	real energyInternal = energyTotal - energyKinetic - energyPotential;
	
	real pressure = (GAMMA - 1.f) * density * energyInternal;

	//von Neumann-Richtmyer artiﬁcial viscosity
	real deltaVelocitySq = 0.f;
	for (int side = 0; side < 2; ++side) {
		int2 iPrev = i;
		--iPrev[side];
		int indexPrev = INDEXGHOSTV(iPrev);
		
		int2 iNext = i;
		++iNext[side];
		int indexNext = INDEXGHOSTV(iNext);

		real velocityL = stateBuffer[indexPrev][side+1];
		real velocityR = stateBuffer[indexNext][side+1];
		const float ZETA = 2.f;
		real deltaVelocity = ZETA * .5f * (velocityR - velocityL);
		deltaVelocitySq += deltaVelocity * deltaVelocity; 
	}
	pressure += deltaVelocitySq * density;

	pressureBuffer[index] = pressure;
}

__kernel void computePressure(
	__global real* pressureBuffer,	//ghost
	const __global real4* stateBuffer,	//ghost
	const __global real* gravityPotentialBuffer,	//ghost
	int2 size)
{
	int2 i = (int2)(get_global_id(0), get_global_id(1));
	computePressureAt(pressureBuffer, stateBuffer, gravityPotentialBuffer, size, i);
}

__kernel void computePressureHorizontal(
	__global real* pressureBuffer,
	const __global real4* stateBuffer,
	const __global real* gravityPotentialBuffer,
	int2 size,
	int y)
{
	int2 i = (int2)(get_global_id(0), y);
	computePressureAt(pressureBuffer, stateBuffer, gravityPotentialBuffer, size, i);
}

__kernel void computePressureVertical(
	__global real* pressureBuffer,
	const __global real4* stateBuffer,
	const __global real* gravityPotentialBuffer,
	int2 size,
	int x)
{
	int2 i = (int2)(x, get_global_id(0));
	computePressureAt(pressureBuffer, stateBuffer, gravityPotentialBuffer, size, i);
}


__kernel void addGravity(
	__global real4* stateBuffer,	//ghost
	const __global real* gravityPotentialBuffer,	//ghost
	int2 size,
	real2 dx,
	const __global real* dtBuffer)
{
	real dt = dtBuffer[0];
	real2 dt_dx = dt / dx;

	int2 i = (int2)(get_global_id(0), get_global_id(1));
	int index = INDEXGHOSTV(i);

	real4 state = stateBuffer[index];
	real density = state.x;

	for (int side = 0; side < 2; ++side) {
		int2 iPrev = i;
		--iPrev[side];
		int indexPrev = INDEXGHOSTV(iPrev);
		
		int2 iNext = i;
		++iNext[side];
		int indexNext = INDEXGHOSTV(iNext);
	
		real gravityGrad = .5f * (gravityPotentialBuffer[indexNext] - gravityPotentialBuffer[indexPrev]);
		
		state[side+1] -= dt_dx[side] * density * gravityGrad;
		state.w -= dt * density * gravityGrad * state[side+1];
	}

	stateBuffer[index] = state;
}

__kernel void diffuseMomentum(
	__global real4* stateBuffer,	//ghost
	const __global real* pressureBuffer,	//ghost
	int2 size,
	real2 dx,
	const __global real* dtBuffer)
{
	real2 dt_dx = dtBuffer[0] / dx;
	
	int2 i = (int2)(get_global_id(0), get_global_id(1));
	int index = INDEXGHOSTV(i);

	for (int side = 0; side < 2; ++side) {
		int2 iPrev = i;
		--iPrev[side];
		int indexPrev = INDEXGHOSTV(iPrev);
		
		int2 iNext = i;
		++iNext[side];
		int indexNext = INDEXGHOSTV(iNext);

		real pressureL = pressureBuffer[indexPrev];
		real pressureR = pressureBuffer[indexNext];

		real deltaPressure = .5f * (pressureR - pressureL);
		stateBuffer[index][side+1] -= deltaPressure * dt_dx[side];
	}
}

__kernel void diffuseWork(
	__global real4* stateBuffer,	//ghost
	const __global real* pressureBuffer,	//ghost
	int2 size,
	real2 dx,
	const __global real* dtBuffer)
{
	real2 dt_dx = dtBuffer[0] / dx;
	
	int2 i = (int2)(get_global_id(0), get_global_id(1));
	int index = INDEXGHOSTV(i);

	for (int side = 0; side < 2; ++side) {
		int2 iPrev = i;
		--iPrev[side];
		int indexPrev = INDEXGHOSTV(iPrev);
		
		int2 iNext = i;
		++iNext[side];
		int indexNext = INDEXGHOSTV(iNext);

		real4 stateL = stateBuffer[indexPrev];
		real4 stateR = stateBuffer[indexNext];

		real velocityL = stateL[side+1] / stateL.x;
		real velocityR = stateR[side+1] / stateR.x;
		
		real pressureL = pressureBuffer[indexPrev];
		real pressureR = pressureBuffer[indexNext];

		real deltaWork = .5f * (pressureR * velocityR - pressureL * velocityL);

		stateBuffer[index].w -= deltaWork * dt_dx[side];
	}
}

