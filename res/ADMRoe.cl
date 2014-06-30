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
	const __global real* gravityPotentialBuffer)
{
	real4 dx = (real4)(DX, DY, DZ, 1.f);

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

#if NUM_STATES != 3
#error only supports 1D 
#endif
	
	for (int side = 0; side < DIM; ++side) {	
		int indexPrev = index - stepsize[side];
		int indexPrev2 = indexPrev - stepsize[side];
		int indexNext = indexNext + stepsize[side];

		int interfaceIndex = side + DIM * index;
		
		const __global real* stateL2 = stateBuffer + NUM_STATES * indexPrev2;
		const __global real* stateL = stateBuffer + NUM_STATES * indexPrev;
		const __global real* stateR = stateBuffer + NUM_STATES * index;
		const __global real* stateR2 = stateBuffer + NUM_STATES * indexNext;
		
		__global real* eigenvalues = eigenvaluesBuffer + NUM_STATES * interfaceIndex;
		__global real* eigenvectors = eigenvectorsBuffer + NUM_STATES * NUM_STATES * interfaceIndex;
		__global real* eigenvectorsInverse = eigenvectorsInverseBuffer + NUM_STATES * NUM_STATES * interfaceIndex;

		//q0 = d/dx ln alpha
		//q1 = d/dx ln g = d/dx ln g_xx
		
		real ln_alphaL = (stateR[STATE_DX_LN_ALPHA] - stateL2[STATE_DX_LN_ALPHA]) / (2.f * dx[side]);
		real alphaL = exp(ln_alphaL);
		real ln_gL = (stateR[STATE_DX_LN_G] - stateL2[STATE_DX_LN_G]) / (2.f * dx[side]);
		real gL = exp(ln_gL);
		real weightL = .5f;
		
		real ln_alphaR = (stateR2[STATE_DX_LN_ALPHA] - stateL[STATE_DX_LN_ALPHA]) / (2.f * dx[side]);
		real alphaR = exp(ln_alphaR);
		real ln_gR = (stateR2[STATE_DX_LN_G] - stateL[STATE_DX_LN_G]) / (2.f * dx[side]);
		real gR = exp(ln_gR);
		real weightR = .5f;
	
		real invDenom = weightL + weightR;
		real alpha = (alphaL * weightL + alphaR * weightR) * invDenom;
		real g = (gL * weightL + gR * weightR) * invDenom;

		const real f = ADM_BONA_MASSO_F;
		
		real sqrtF = sqrt(f);
		real oneOverF = 1.f / f;
		real oneOverSqrtF = sqrt(oneOverF);
		
		real sqrtG = sqrt(g);
		real oneOverSqrtG = 1.f / sqrtG;	
		
		//eigenvalues

		real eigenvalue = alpha * sqrtF * oneOverSqrtG;
		eigenvalues[0] = -eigenvalue;
		eigenvalues[1] = 0.f;
		eigenvalues[2] = eigenvalue;

		//eigenvectors

		//col
		
		eigenvectors[0 + NUM_STATES * 0] = f;
		eigenvectors[1 + NUM_STATES * 0] = 2.f; 
		eigenvectors[2 + NUM_STATES * 0] = -sqrtF;
		//col
		eigenvectors[0 + NUM_STATES * 1] = 0.f;
		eigenvectors[1 + NUM_STATES * 1] = 1.f;
		eigenvectors[2 + NUM_STATES * 1] = 0.f;
		//col
		eigenvectors[0 + NUM_STATES * 2] = f;
		eigenvectors[1 + NUM_STATES * 2] = 2.f;
		eigenvectors[2 + NUM_STATES * 2] = sqrtF;

		//calculate eigenvector inverses ... 
		//min row
		eigenvectorsInverse[0 + NUM_STATES * 0] = .5f * oneOverF;
		eigenvectorsInverse[0 + NUM_STATES * 1] = 0.f;
		eigenvectorsInverse[0 + NUM_STATES * 2] = -.5f * oneOverSqrtF;
		//mid normal row
		eigenvectorsInverse[1 + NUM_STATES * 0] = -2.f * oneOverF;
		eigenvectorsInverse[1 + NUM_STATES * 1] = 1.f;
		eigenvectorsInverse[1 + NUM_STATES * 2] = 0.f;
		//mid tangent A row
		eigenvectorsInverse[2 + NUM_STATES * 0] = .5f * oneOverF;
		eigenvectorsInverse[2 + NUM_STATES * 1] = 0.f;
		eigenvectorsInverse[2 + NUM_STATES * 2] = .5f * oneOverSqrtF;
	}
}

