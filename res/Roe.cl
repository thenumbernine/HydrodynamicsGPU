#include "HydroGPU/Shared/Common.h"
#include "HydroGPU/Roe.h"

__kernel void findMinTimestep(
	__global real* dtBuffer,
	const __global real* eigenvaluesBuffer
#ifdef SOLID
	, const __global char* solidBuffer
#endif	//SOLID
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
		dtBuffer[index] = INFINITY;
		return;
	}
	int indexL = index;

	real result = INFINITY;
	for (int side = 0; side < DIM; ++side) {
		int indexR = index + stepsize[side];

#ifdef SOLID
		if (solidBuffer[indexL] || solidBuffer[indexR]) continue;
#endif	//SOLID

		const __global real* eigenvaluesL = eigenvaluesBuffer + EIGEN_SPACE_DIM * (side + DIM * indexL);
		const __global real* eigenvaluesR = eigenvaluesBuffer + EIGEN_SPACE_DIM * (side + DIM * indexR);
		
		//NOTICE assumes eigenvalues are sorted from min to max
		real maxLambda = max(0.f, eigenvaluesL[EIGEN_SPACE_DIM-1]);
		real minLambda = min(0.f, eigenvaluesR[0]);
		
		real dum = dx[side] / (fabs(maxLambda - minLambda) + 1e-9f);
		result = min(result, dum);
	}
	
	dtBuffer[index] = result;
}

__kernel void calcDeltaQTilde(
	__global real* deltaQTildeBuffer,
	const __global real* eigenfieldsBuffer,
	const __global real* stateBuffer,
	int side
#ifdef SOLID
	, const __global char* solidBuffer
#endif	//SOLID
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
#endif	//SOLID

#ifdef ROE_EIGENFIELD_TRANSFORM_SEPARATE
	//calculating this twice because eigenfieldTransform could use the state variables to construct the field information on the fly

	real stateLTilde[EIGEN_SPACE_DIM];
	eigenfieldTransform(stateLTilde, eigenfields, stateL, side);

	real stateRTilde[EIGEN_SPACE_DIM];
	eigenfieldTransform(stateRTilde, eigenfields, stateR, side);

	for (int i = 0; i < EIGEN_SPACE_DIM; ++i) {
		deltaQTilde[i] = stateRTilde[i] - stateLTilde[i];
	}
#else	//ROE_EIGENFIELD_TRANSFORM_SEPARATE
	real deltaState[NUM_STATES];
	real deltaQTilde_[EIGEN_SPACE_DIM];
	for (int i = 0; i < NUM_STATES; ++i) {
		deltaState[i] = stateR[i] - stateL[i];
	}
	eigenfieldTransform(deltaQTilde_, eigenfields, deltaState, side);
	for (int i = 0; i < EIGEN_SPACE_DIM; ++i) {
		deltaQTilde[i] = deltaQTilde_[i];
	}
#endif	//ROE_EIGENFIELD_TRANSFORM_SEPARATE
}

__kernel void calcFlux(
	__global real* fluxBuffer,
	const __global real* stateBuffer,
	const __global real* eigenvaluesBuffer,
	const __global real* eigenfieldsBuffer,
	const __global real* deltaQTildeBuffer,
	real dt,
	int side
#ifdef SOLID
	, const __global char* solidBuffer
#endif	//SOLID
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
	
	real dt_dx = dt / dx[side];

	int index = INDEXV(i);
	int indexR = index;	
	
	int indexL = index - stepsize[side];
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
	int indexL2 = indexL - stepsize[side];
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
#endif	//SOLID

	real fluxTilde[EIGEN_SPACE_DIM];
#ifdef ROE_EIGENFIELD_TRANSFORM_SEPARATE
	real stateLTilde[EIGEN_SPACE_DIM];
	eigenfieldTransform(stateLTilde, eigenfields, stateL, side);

	real stateRTilde[EIGEN_SPACE_DIM];
	eigenfieldTransform(stateRTilde, eigenfields, stateR, side);

	for (int i = 0; i < EIGEN_SPACE_DIM; ++i) {
		fluxTilde[i] = .5f * (stateRTilde[i] + stateLTilde[i]);
	}
#else	//ROE_EIGENFIELD_TRANSFORM_SEPARATE
	real stateAvg[NUM_STATES];
	for (int i = 0; i < NUM_STATES; ++i) {
		stateAvg[i] = .5f * (stateR[i] + stateL[i]);
	}
	eigenfieldTransform(fluxTilde, eigenfields, stateAvg, side);
#endif	//ROE_EIGENFIELD_TRANSFORM_SEPARATE

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
#endif	//SOLID
		} else {
			rTilde = deltaQTildeR[i] / deltaQTilde[i];
			theta = -1.f;
#ifdef SOLID
			if (solidR2) rTilde = 1.f;
#endif	//SOLID
		}
		real phi = slopeLimiter(rTilde);
		real epsilon = eigenvalue * dt_dx;

		//TODO some research on >1D flux limiters
		//this makes things stable, but I still haven't figured out why
//		phi /= DIM;

		real deltaFluxTilde = eigenvalue * deltaQTilde[i];
		fluxTilde[i] -= .5f * deltaFluxTilde * (theta + phi * (epsilon - theta));
	}

	eigenfieldInverseTransform(flux, eigenfields, fluxTilde, side);
}

__kernel void calcFluxDeriv(
	__global real* derivBuffer,
	const __global real* fluxBuffer,
	int side
#ifdef SOLID
	, const __global char* solidBuffer
#endif	//SOLID
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
#endif	//SOLID

	__global real* deriv = derivBuffer + NUM_STATES * index;

	int indexNext = index + stepsize[side];
	const __global real* fluxL = fluxBuffer + NUM_STATES * (side + DIM * index); 
	const __global real* fluxR = fluxBuffer + NUM_STATES * (side + DIM * indexNext); 
	for (int j = 0; j < NUM_STATES; ++j) {
		real deltaFlux = fluxR[j] - fluxL[j];
		deriv[j] -= deltaFlux / dx[side];
	}
}

