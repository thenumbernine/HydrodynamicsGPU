#include "HydroGPU/Shared/Common.h"
#include "HydroGPU/Roe.h"

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
	
		//NOTICE assumes eigenvalues are sorted from min to max
		real maxLambda = max(0.f, eigenvaluesL[NUM_STATES-1]);
		real minLambda = min(0.f, eigenvaluesR[0]);
		
		real dum = dx[side] / (fabs(maxLambda - minLambda) + 1e-9f);
		result = min(result, dum);
	}
	
	cflBuffer[index] = cfl * result;
}

void calcDeltaQTildeSide(
	__global real* deltaQTildeBuffer,
	const __global real* eigenfieldsBuffer,
	const __global real* stateBuffer,
	int side);

void calcDeltaQTildeSide(
	__global real* deltaQTildeBuffer,
	const __global real* eigenfieldsBuffer,
	const __global real* stateBuffer,
	int side)
{
	int4 i = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);
	int index = INDEXV(i);
	int indexPrev = index - stepsize[side];
	int interfaceIndex = side + DIM * index;
			
	const __global real* stateL = stateBuffer + NUM_STATES * indexPrev;
	const __global real* stateR = stateBuffer + NUM_STATES * index;
	const __global real* eigenfields = eigenfieldsBuffer + EIGENFIELD_SIZE * interfaceIndex;
	__global real* deltaQTilde = deltaQTildeBuffer + NUM_STATES * interfaceIndex;

	real deltaQ[NUM_STATES];
	for (int i = 0; i < NUM_STATES; ++i) {
		deltaQ[i] = stateR[i] - stateL[i];
	}
	real deltaQTilde_[NUM_STATES];
	eigenfieldTransform(deltaQTilde_, eigenfields, deltaQ);
	for (int i = 0; i < NUM_STATES; ++i) {
		deltaQTilde[i] = deltaQTilde_[i];
	}
}

__kernel void calcDeltaQTilde(
	__global real* deltaQTildeBuffer,
	const __global real* eigenfieldsBuffer,
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
	calcDeltaQTildeSide(deltaQTildeBuffer, eigenfieldsBuffer, stateBuffer, 0);
#if DIM > 1
	calcDeltaQTildeSide(deltaQTildeBuffer, eigenfieldsBuffer, stateBuffer, 1);
#endif
#if DIM > 2
	calcDeltaQTildeSide(deltaQTildeBuffer, eigenfieldsBuffer, stateBuffer, 2);
#endif
}

void calcFluxSide(
	__global real* fluxBuffer,
	const __global real* stateBuffer,
	const __global real* eigenvaluesBuffer,
	const __global real* eigenfieldsInverseBuffer,
	const __global real* eigenfieldsBuffer,
	const __global real* deltaQTildeBuffer,
	real dt_dx,
	int side);

void calcFluxSide(
	__global real* fluxBuffer,
	const __global real* stateBuffer,
	const __global real* eigenvaluesBuffer,
	const __global real* eigenfieldsInverseBuffer,
	const __global real* eigenfieldsBuffer,
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
	const __global real* eigenfieldsInverse = eigenfieldsInverseBuffer + EIGENFIELD_SIZE * interfaceIndex;
	const __global real* eigenfields = eigenfieldsBuffer + EIGENFIELD_SIZE * interfaceIndex;
	__global real* flux = fluxBuffer + NUM_STATES * interfaceIndex;

	
	real avgQ[NUM_STATES];
	for (int i = 0; i < NUM_STATES; ++i) {
		avgQ[i] = .5f * (stateR[i] + stateL[i]);
	}
	real fluxTilde[NUM_STATES];
	eigenfieldTransform(fluxTilde, eigenfields, avgQ);	

	for (int i = 0; i < NUM_STATES; ++i) {
		real eigenvalue = eigenvalues[i];
		fluxTilde[i] *= eigenvalue;

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

	eigenfieldInverseTransform(flux, eigenfieldsInverse, fluxTilde);
}

__kernel void calcFlux(
	__global real* fluxBuffer,
	const __global real* stateBuffer,
	const __global real* eigenvaluesBuffer,
	const __global real* eigenfieldsInverseBuffer,
	const __global real* eigenfieldsBuffer,
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
	calcFluxSide(fluxBuffer, stateBuffer, eigenvaluesBuffer, eigenfieldsInverseBuffer, eigenfieldsBuffer, deltaQTildeBuffer, dt / DX, 0); 
#if DIM > 1
	calcFluxSide(fluxBuffer, stateBuffer, eigenvaluesBuffer, eigenfieldsInverseBuffer, eigenfieldsBuffer, deltaQTildeBuffer, dt / DY, 1); 
#endif
#if DIM > 2
	calcFluxSide(fluxBuffer, stateBuffer, eigenvaluesBuffer, eigenfieldsInverseBuffer, eigenfieldsBuffer, deltaQTildeBuffer, dt / DZ, 2); 
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

