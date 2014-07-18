#include "HydroGPU/Shared/Common.h"

__kernel void calcCFL(
	__global real* cflBuffer,
	const __global real* eigenvaluesBuffer,
	real cfl)
{
	int4 i = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);
	int index = INDEXV(i);
	if (i.x < 2 || i.x >= SIZE_X - 2 
#if DIM > 1
		|| i.y < 2 || i.y >= SIZE_Y - 2
#endif
#if DIM > 2
		|| i.z < 2 || i.z >= SIZE_Z - 2
#endif
	) {
		cflBuffer[index] = INFINITY;
		return;
	}
	int indexL = index;

	real result = INFINITY;
	for (int side = 0; side < DIM; ++side) {
		int indexR = index + stepsize[side];
		
		const __global real* eigenvaluesL = eigenvaluesBuffer + NUM_STATES * (side + DIM * indexL);
		const __global real* eigenvaluesR = eigenvaluesBuffer + NUM_STATES * (side + DIM * indexR);
	
		real minLambda = 0.f;
		real maxLambda = 0.f;
		for (int i = 0; i < NUM_STATES; ++i) {	
			maxLambda = max(maxLambda, eigenvaluesL[i]);
			minLambda = min(minLambda, eigenvaluesR[i]);
		}

		real dum = dx[side] / (fabs(maxLambda - minLambda) + 1e-9f);
		result = min(result, dum);
	}
		
	cflBuffer[index] = cfl * result;
}

void calcDeltaQTildeSide(
	__global real* deltaQTildeBuffer,
	const __global real* eigenvectorsInverseBuffer,
	const __global real* stateBuffer,
	int side);

void calcDeltaQTildeSide(
	__global real* deltaQTildeBuffer,
	const __global real* eigenvectorsInverseBuffer,
	const __global real* stateBuffer,
	int side)
{
	int4 i = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);
	int index = INDEXV(i);
	int indexPrev = index - stepsize[side];
	int interfaceIndex = side + DIM * index;
			
	const __global real* stateL = stateBuffer + NUM_STATES * indexPrev;
	const __global real* stateR = stateBuffer + NUM_STATES * index;
	const __global real* eigenvectorsInverse = eigenvectorsInverseBuffer + NUM_STATES * NUM_STATES * interfaceIndex;
	__global real* deltaQTilde = deltaQTildeBuffer + NUM_STATES * interfaceIndex;

	for (int i = 0; i < NUM_STATES; ++i) {
		real sum = 0.f;
		for (int j = 0; j < NUM_STATES; ++j) {
			sum += eigenvectorsInverse[i + NUM_STATES * j] * (stateR[j] - stateL[j]);
		}
		deltaQTilde[i] = sum;
	}
}

__kernel void calcDeltaQTilde(
	__global real* deltaQTildeBuffer,
	const __global real* eigenvectorsInverseBuffer,
	const __global real* stateBuffer)
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
	calcDeltaQTildeSide(deltaQTildeBuffer, eigenvectorsInverseBuffer, stateBuffer, 0);
#if DIM > 1
	calcDeltaQTildeSide(deltaQTildeBuffer, eigenvectorsInverseBuffer, stateBuffer, 1);
#endif
#if DIM > 2
	calcDeltaQTildeSide(deltaQTildeBuffer, eigenvectorsInverseBuffer, stateBuffer, 2);
#endif
}

void calcFluxSide(
	__global real* fluxBuffer,
	const __global real* stateBuffer,
	const __global real* eigenvaluesBuffer,
	const __global real* eigenvectorsBuffer,
	const __global real* eigenvectorsInverseBuffer,
	const __global real* deltaQTildeBuffer,
	real dt_dx,
	int side);

void calcFluxSide(
	__global real* fluxBuffer,
	const __global real* stateBuffer,
	const __global real* eigenvaluesBuffer,
	const __global real* eigenvectorsBuffer,
	const __global real* eigenvectorsInverseBuffer,
	const __global real* deltaQTildeBuffer,
	real dt_dx,
	int side)
{
	int4 i = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);
	int index = INDEXV(i);
	int indexR = index;	
	
	int indexL = index - stepsize[side];
	int indexR2 = index + stepsize[side];

	int interfaceLIndex = side + DIM * indexL;
	int interfaceIndex = side + DIM * indexR;
	int interfaceRIndex = side + DIM * indexR2;
	
	const __global real* stateL = stateBuffer + NUM_STATES * indexL;
	const __global real* stateR = stateBuffer + NUM_STATES * indexR;
	
	const __global real* deltaQTildeL = deltaQTildeBuffer + NUM_STATES * interfaceLIndex;
	const __global real* deltaQTilde = deltaQTildeBuffer + NUM_STATES * interfaceIndex;
	const __global real* deltaQTildeR = deltaQTildeBuffer + NUM_STATES * interfaceRIndex;
	
	const __global real* eigenvalues = eigenvaluesBuffer + NUM_STATES * interfaceIndex;
	const __global real* eigenvectors = eigenvectorsBuffer + NUM_STATES * NUM_STATES * interfaceIndex;
	const __global real* eigenvectorsInverse = eigenvectorsInverseBuffer + NUM_STATES * NUM_STATES * interfaceIndex;
	__global real* flux = fluxBuffer + NUM_STATES * interfaceIndex;

	real fluxTilde[NUM_STATES];
	for (int i = 0; i < NUM_STATES; ++i) {
		real eigenvalue = eigenvalues[i];
		
		real sum = 0.f;
		for (int j = 0; j < NUM_STATES; ++j) {
			sum += eigenvectorsInverse[i + NUM_STATES * j] * (stateR[j] + stateL[j]);
		}
		fluxTilde[i] = .5f * sum * eigenvalue;

		real rTilde;
		real theta;
		if (eigenvalue >= 0.f) {
			rTilde = deltaQTildeL[i] / deltaQTilde[i];
			theta = 1.f;
		} else {
			rTilde = deltaQTildeR[i] / deltaQTilde[i];
			theta = -1.f;
		}
		real phi = slopeLimiter(rTilde);
		real epsilon = eigenvalue * dt_dx;

		real deltaFluxTilde = eigenvalue * deltaQTilde[i];
		fluxTilde[i] -= .5f * deltaFluxTilde * (theta + phi * (epsilon - theta) / (float)DIM);
	}

	for (int i = 0; i < NUM_STATES; ++i) {
		real sum = 0.f;
		for (int j = 0; j < NUM_STATES; ++j) {
			sum += eigenvectors[i + NUM_STATES * j] * fluxTilde[j];
		}
		flux[i] = sum;
	}
}

__kernel void calcFlux(
	__global real* fluxBuffer,
	const __global real* stateBuffer,
	const __global real* eigenvaluesBuffer,
	const __global real* eigenvectorsBuffer,
	const __global real* eigenvectorsInverseBuffer,
	const __global real* deltaQTildeBuffer,
	const __global real* dtBuffer)
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
	
	float dt = dtBuffer[0];
	calcFluxSide(fluxBuffer, stateBuffer, eigenvaluesBuffer, eigenvectorsBuffer, eigenvectorsInverseBuffer, deltaQTildeBuffer, dt / DX, 0); 
#if DIM > 1
	calcFluxSide(fluxBuffer, stateBuffer, eigenvaluesBuffer, eigenvectorsBuffer, eigenvectorsInverseBuffer, deltaQTildeBuffer, dt / DY, 1); 
#endif
#if DIM > 2
	calcFluxSide(fluxBuffer, stateBuffer, eigenvaluesBuffer, eigenvectorsBuffer, eigenvectorsInverseBuffer, deltaQTildeBuffer, dt / DZ, 2); 
#endif
}

__kernel void calcFluxDeriv(
	__global real* derivBuffer,
	const __global real* fluxBuffer)
{
	int4 i = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);
	if (i.x < 2 || i.x >= SIZE_X - 2 
#if DIM > 1
		|| i.y < 2 || i.y >= SIZE_Y - 2 
#endif
#if DIM > 2
		|| i.z < 2 || i.z >= SIZE_Z - 2
#endif
	) {
		return;
	}
	int index = INDEXV(i);
	
	__global real* deriv = derivBuffer + NUM_STATES * index;

	for (int side = 0; side < DIM; ++side) {
		int indexNext = index + stepsize[side];
		const __global real* fluxL = fluxBuffer + NUM_STATES * (side + DIM * index); 
		const __global real* fluxR = fluxBuffer + NUM_STATES * (side + DIM * indexNext); 
		for (int j = 0; j < NUM_STATES; ++j) {
			real deltaFlux = fluxR[j] - fluxL[j];
			deriv[j] -= deltaFlux / dx[side];
		}
	}
}

