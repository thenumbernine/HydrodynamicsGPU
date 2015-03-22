#include "HydroGPU/Shared/Common.h"
#include "HydroGPU/Roe.h"

__kernel void calcCFL(
	__global real* cflBuffer,
	const __global real* eigenvaluesBuffer,
	real cfl
#ifdef SOLID
	, const __global char* solidBuffer
#endif
)
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

#ifdef SOLID
		if (solidBuffer[indexL] || solidBuffer[indexR]) continue;
#endif

		const __global real* eigenvaluesL = eigenvaluesBuffer + EIGEN_SPACE_DIM * (side + DIM * indexL);
		const __global real* eigenvaluesR = eigenvaluesBuffer + EIGEN_SPACE_DIM * (side + DIM * indexR);
		
		//NOTICE assumes eigenvalues are sorted from min to max
		real maxLambda = max(0.f, eigenvaluesL[EIGEN_SPACE_DIM-1]);
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
	int side
#ifdef SOLID
	, const __global char* solidBuffer
#endif	
	);

void calcDeltaQTildeSide(
	__global real* deltaQTildeBuffer,
	const __global real* eigenfieldsBuffer,
	const __global real* stateBuffer,
	int side
#ifdef SOLID
	, const __global char* solidBuffer
#endif
)
{
	int4 i = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);
	int index = INDEXV(i);
	int indexPrev = index - stepsize[side];
	int interfaceIndex = side + DIM * index;
			
	const __global real* eigenfields = eigenfieldsBuffer + EIGEN_TRANSFORM_STRUCT_SIZE * interfaceIndex;
	__global real* deltaQTilde = deltaQTildeBuffer + EIGEN_SPACE_DIM * interfaceIndex;

	real stateL[NUM_STATES];
	real stateR[NUM_STATES];
	for (int i = 0; i < NUM_STATES; ++i) {
		stateL[i] = stateBuffer[i + NUM_STATES * indexPrev];
		stateR[i] = stateBuffer[i + NUM_STATES * index];
	}
#ifdef SOLID
	char solidL = solidBuffer[indexPrev];
	char solidR = solidBuffer[index];
	if (solidL && !solidR) {
		for (int i = 0; i < NUM_STATES; ++i) {
			stateL[i] = stateR[i];
		}
		stateL[side+STATE_MOMENTUM_X] = -stateL[side+STATE_MOMENTUM_X];
	} else if (solidR && !solidL) {
		for (int i = 0; i < NUM_STATES; ++i) {
			stateR[i] = stateL[i];
		}
		stateR[side+STATE_MOMENTUM_X] = -stateR[side+STATE_MOMENTUM_X];
	}
#endif

	//calculating this twice because eigenfieldTransform could use the state variables to construct the field information on the fly

	real stateLTilde[EIGEN_SPACE_DIM];
	eigenfieldTransform(stateLTilde, eigenfields, stateL, side);

	real stateRTilde[EIGEN_SPACE_DIM];
	eigenfieldTransform(stateRTilde, eigenfields, stateR, side);

	for (int i = 0; i < EIGEN_SPACE_DIM; ++i) {
		deltaQTilde[i] = stateRTilde[i] - stateLTilde[i];
	}
}

__kernel void calcDeltaQTilde(
	__global real* deltaQTildeBuffer,
	const __global real* eigenfieldsBuffer,
	const __global real* stateBuffer
#ifdef SOLID
	, const __global char* solidBuffer
#endif
)
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
	calcDeltaQTildeSide(deltaQTildeBuffer, eigenfieldsBuffer, stateBuffer, 0
#ifdef SOLID
		, solidBuffer
#endif
	);
#if DIM > 1
	calcDeltaQTildeSide(deltaQTildeBuffer, eigenfieldsBuffer, stateBuffer, 1
#ifdef SOLID
		, solidBuffer
#endif
	);
#endif
#if DIM > 2
	calcDeltaQTildeSide(deltaQTildeBuffer, eigenfieldsBuffer, stateBuffer, 2
#ifdef SOLID
		, solidBuffer
#endif
	);
#endif
}

void calcFluxSide(
	__global real* fluxBuffer,
	const __global real* stateBuffer,
	const __global real* eigenvaluesBuffer,
	const __global real* eigenfieldsBuffer,
	const __global real* deltaQTildeBuffer,
	real dt_dx,
	int side
#ifdef SOLID
	, const __global char* solidBuffer
#endif
	);

void calcFluxSide(
	__global real* fluxBuffer,
	const __global real* stateBuffer,
	const __global real* eigenvaluesBuffer,
	const __global real* eigenfieldsBuffer,
	const __global real* deltaQTildeBuffer,
	real dt_dx,
	int side
#ifdef SOLID
	, const __global char* solidBuffer
#endif
)
{
	int4 i = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);
	int index = INDEXV(i);
	int indexR = index;	
	
	int indexL = index - stepsize[side];
	int indexL2 = indexL - stepsize[side];
	int indexR2 = indexR + stepsize[side];

	int interfaceLIndex = side + DIM * indexL;
	int interfaceIndex = side + DIM * indexR;
	int interfaceRIndex = side + DIM * indexR2;
	
	const __global real* deltaQTildeL = deltaQTildeBuffer + EIGEN_SPACE_DIM * interfaceLIndex;
	const __global real* deltaQTilde = deltaQTildeBuffer + EIGEN_SPACE_DIM * interfaceIndex;
	const __global real* deltaQTildeR = deltaQTildeBuffer + EIGEN_SPACE_DIM * interfaceRIndex;
	
	const __global real* eigenvalues = eigenvaluesBuffer + EIGEN_SPACE_DIM * interfaceIndex;
	const __global real* eigenfields = eigenfieldsBuffer + EIGEN_TRANSFORM_STRUCT_SIZE * interfaceIndex;
	__global real* flux = fluxBuffer + NUM_STATES * interfaceIndex;

	real stateL[NUM_STATES];
	for (int i = 0; i < NUM_STATES; ++i) {
		stateL[i] = stateBuffer[i + NUM_STATES * indexL];
	}
	real stateR[NUM_STATES];
	for (int i = 0; i < NUM_STATES; ++i) {
		stateR[i] = stateBuffer[i + NUM_STATES * indexR];
	}
#ifdef SOLID
	char solidL = solidBuffer[indexL];
	char solidR = solidBuffer[indexR];
	if (solidL && !solidR) {
		for (int i = 0; i < NUM_STATES; ++i) {
			stateL[i] = stateR[i];
		}
		stateL[side+STATE_MOMENTUM_X] = -stateL[side+STATE_MOMENTUM_X];
	} else if (solidR && !solidL) {
		for (int i = 0; i < NUM_STATES; ++i) {
			stateR[i] = stateL[i];
		}
		stateR[side+STATE_MOMENTUM_X] = -stateR[side+STATE_MOMENTUM_X];
	}
	char solidL2 = solidBuffer[indexL2];
	char solidR2 = solidBuffer[indexR2];
#endif
	
	real stateLTilde[EIGEN_SPACE_DIM];
	eigenfieldTransform(stateLTilde, eigenfields, stateL, side);

	real stateRTilde[EIGEN_SPACE_DIM];
	eigenfieldTransform(stateRTilde, eigenfields, stateR, side);

	real fluxTilde[EIGEN_SPACE_DIM];
	for (int i = 0; i < EIGEN_SPACE_DIM; ++i) {
		fluxTilde[i] = .5f * (stateRTilde[i] + stateLTilde[i]);
	}

	for (int i = 0; i < EIGEN_SPACE_DIM; ++i) {
		real eigenvalue = eigenvalues[i];
		fluxTilde[i] *= eigenvalue;

		real rTilde;
		real theta;
		if (eigenvalue >= 0.f) {
			rTilde = deltaQTildeL[i] / deltaQTilde[i];
			theta = 1.f;
#ifdef SOLID
			if (solidL2) rTilde = 1.f;
#endif
		} else {
			rTilde = deltaQTildeR[i] / deltaQTilde[i];
			theta = -1.f;
#ifdef SOLID
			if (solidR2) rTilde = 1.f;
#endif
		}
		real phi = slopeLimiter(rTilde);
		real epsilon = eigenvalue * dt_dx;

		real deltaFluxTilde = eigenvalue * deltaQTilde[i];
		fluxTilde[i] -= .5f * deltaFluxTilde * (theta + phi * (epsilon - theta)
			// / (float)DIM	//?
		);
	}

	eigenfieldInverseTransform(flux, eigenfields, fluxTilde, side);
}

__kernel void calcFlux(
	__global real* fluxBuffer,
	const __global real* stateBuffer,
	const __global real* eigenvaluesBuffer,
	const __global real* eigenfieldsBuffer,
	const __global real* deltaQTildeBuffer,
	const __global real* dtBuffer
#ifdef SOLID
	, const __global char* solidBuffer
#endif
)
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
	calcFluxSide(fluxBuffer, stateBuffer, eigenvaluesBuffer, eigenfieldsBuffer, deltaQTildeBuffer, dt / DX, 0
#ifdef SOLID
	, solidBuffer
#endif
	); 
#if DIM > 1
	calcFluxSide(fluxBuffer, stateBuffer, eigenvaluesBuffer, eigenfieldsBuffer, deltaQTildeBuffer, dt / DY, 1
#ifdef SOLID
	, solidBuffer
#endif
	); 
#endif
#if DIM > 2
	calcFluxSide(fluxBuffer, stateBuffer, eigenvaluesBuffer, eigenfieldsBuffer, deltaQTildeBuffer, dt / DZ, 2
#ifdef SOLID
	, solidBuffer
#endif
	); 
#endif
}

__kernel void calcFluxDeriv(
	__global real* derivBuffer,
	const __global real* fluxBuffer
#ifdef SOLID
	, const __global char* solidBuffer
#endif
	)
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

#ifdef SOLID
	if (solidBuffer[index]) return;
#endif

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

