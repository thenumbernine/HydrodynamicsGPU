/*
The components of the Roe solver specific to the ADM equations
paritcularly the spectral decomposition

This currently only supports 1D
*/

#include "HydroGPU/Shared/Common.h"

__kernel void calcEigenBasis(
	__global real* eigenvaluesBuffer,
	__global real* eigenvectorsBuffer,
	__global real* eigenvectorsInverseBuffer,
	const __global real* stateBuffer,
	const __global real* potentialBuffer)
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

#if NUM_STATES != 5
#error only supports 1D 
#endif
	
	//for (int side = 0; side < DIM; ++side) {
	{const int side = 0;
		int indexPrev = index - stepsize[side];
		int indexNext = indexNext + stepsize[side];

		int interfaceIndex = side + DIM * index;
		
		const __global real* stateL = stateBuffer + NUM_STATES * indexPrev;
		const __global real* stateR = stateBuffer + NUM_STATES * index;
		
		__global real* eigenvalues = eigenvaluesBuffer + NUM_STATES * interfaceIndex;
		__global real* eigenvectors = eigenvectorsBuffer + NUM_STATES * NUM_STATES * interfaceIndex;
		__global real* eigenvectorsInverse = eigenvectorsInverseBuffer + NUM_STATES * NUM_STATES * interfaceIndex;

		//q0 = d/dx ln alpha
		//q1 = d/dx ln g = d/dx ln g_xx
	
		real alpha = .5f * (stateL[STATE_ALPHA] + stateR[STATE_ALPHA]);
		real g = .5f * (stateL[STATE_G] + stateR[STATE_G]);
		real A = .5f * (stateL[STATE_A] + stateR[STATE_A]);
		//real D = .5f * (stateL[STATE_D] + stateR[STATE_D]);
		real K = .5f * (stateL[STATE_K] + stateR[STATE_K]);
		
		const real f = ADM_BONA_MASSO_F;
		
		//eigenvalues

		real eigenvalue = alpha * sqrt(f/g); 
		eigenvalues[0] = -eigenvalue;
		eigenvalues[1] = 0.f;
		eigenvalues[2] = 0.f;
		eigenvalues[3] = 0.f;
		eigenvalues[4] = eigenvalue;

		//eigenvectors

		//col
		eigenvectors[0 + NUM_STATES * 0] = 0.f;
		eigenvectors[1 + NUM_STATES * 0] = 0.f; 
		eigenvectors[2 + NUM_STATES * 0] = f/g; 
		eigenvectors[3 + NUM_STATES * 0] = 1.f; 
		eigenvectors[4 + NUM_STATES * 0] = -sqrt(f/g); 
		//col
		eigenvectors[0 + NUM_STATES * 1] = alpha;
		eigenvectors[1 + NUM_STATES * 1] = 0.f;
		eigenvectors[2 + NUM_STATES * 1] = -A;
		eigenvectors[3 + NUM_STATES * 1] = 0.f;
		eigenvectors[4 + NUM_STATES * 1] = -K;
		//col
		eigenvectors[0 + NUM_STATES * 2] = 0.f;
		eigenvectors[1 + NUM_STATES * 2] = 0.f;
		eigenvectors[2 + NUM_STATES * 2] = 0.f;
		eigenvectors[3 + NUM_STATES * 2] = 1.f;
		eigenvectors[4 + NUM_STATES * 2] = 0.f;
		//col
		eigenvectors[0 + NUM_STATES * 3] = 0.f;
		eigenvectors[1 + NUM_STATES * 3] = 1.f;
		eigenvectors[2 + NUM_STATES * 3] = 0.f;
		eigenvectors[3 + NUM_STATES * 3] = 0.f;
		eigenvectors[4 + NUM_STATES * 3] = 0.f;
		//col
		eigenvectors[0 + NUM_STATES * 4] = 0.f;
		eigenvectors[1 + NUM_STATES * 4] = 0.f;
		eigenvectors[2 + NUM_STATES * 4] = f/g;
		eigenvectors[3 + NUM_STATES * 4] = 1.f;
		eigenvectors[4 + NUM_STATES * 4] = sqrt(f/g);

		//calculate eigenvector inverses ... 
		//min 
		eigenvectorsInverse[0 + NUM_STATES * 0] = (g * A / f - K * sqrt(g / f)) / (2.f * alpha); 
		eigenvectorsInverse[0 + NUM_STATES * 1] = 0.f; 
		eigenvectorsInverse[0 + NUM_STATES * 2] = g / (2.f * f); 
		eigenvectorsInverse[0 + NUM_STATES * 3] = 0.f; 
		eigenvectorsInverse[0 + NUM_STATES * 4] = -.5f * sqrt(g / f); 
		//row
		eigenvectorsInverse[1 + NUM_STATES * 0] = 1.f / alpha;
		eigenvectorsInverse[1 + NUM_STATES * 1] = 0.f;
		eigenvectorsInverse[1 + NUM_STATES * 2] = 0.f;
		eigenvectorsInverse[1 + NUM_STATES * 3] = 0.f;
		eigenvectorsInverse[1 + NUM_STATES * 4] = 0.f;
		//row
		eigenvectorsInverse[2 + NUM_STATES * 0] = -(g * A) / (alpha * f); 
		eigenvectorsInverse[2 + NUM_STATES * 1] = 0.f; 
		eigenvectorsInverse[2 + NUM_STATES * 2] = -g / f; 
		eigenvectorsInverse[2 + NUM_STATES * 3] = 1.f; 
		eigenvectorsInverse[2 + NUM_STATES * 4] = 0.f; 
		//row
		eigenvectorsInverse[3 + NUM_STATES * 0] = 0.f;
		eigenvectorsInverse[3 + NUM_STATES * 1] = 1.f;
		eigenvectorsInverse[3 + NUM_STATES * 2] = 0.f;
		eigenvectorsInverse[3 + NUM_STATES * 3] = 0.f;
		eigenvectorsInverse[3 + NUM_STATES * 4] = 0.f;
		//row
		eigenvectorsInverse[4 + NUM_STATES * 0] = (g * A / f + K * sqrt(g / f)) / (2.f * alpha); 
		eigenvectorsInverse[4 + NUM_STATES * 1] = 0.f; 
		eigenvectorsInverse[4 + NUM_STATES * 2] = g / (2.f * f); 
		eigenvectorsInverse[4 + NUM_STATES * 3] = 0.f; 
		eigenvectorsInverse[4 + NUM_STATES * 4] = .5f * sqrt(g / f); 
	}
}

__kernel void addSource(
	__global real* derivBuffer,
	const __global real* stateBuffer)
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
	const __global real* state = stateBuffer + NUM_STATES * index;

	real alpha = state[0];
	real g = state[1];
	real A = state[2];
	real D = state[3];
	real K = state[4];
	real f = ADM_BONA_MASSO_F;
	deriv[0] += -alpha * alpha * f * K / g;
	deriv[1] += -2.f * alpha * K;
	deriv[4] += + alpha * (A * D - K * K) / g;
}
