/*
The components of the Roe solver specific to the ADM equations
paritcularly the spectral decomposition

This currently only supports 1D
*/

#include "HydroGPU/Shared/Common.h"

__kernel void calcEigenBasis(
	__global real8* eigenvaluesBuffer,
	__global real16* eigenvectorsBuffer,
	__global real16* eigenvectorsInverseBuffer,
	const __global real8* stateBuffer,
	const __global real* gravityPotentialBuffer)
{
	real4 dx = (real4)(DX, DY, DZ, 1.);

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

	for (int side = 0; side < DIM; ++side) {	
		int4 iPrev = i;
		--iPrev[side];
		int indexPrev = INDEXV(iPrev);

		int4 iPrev2 = iPrev;
		--iPrev2[side];
		int indexPrev2 = INDEXV(iPrev2);

		int4 iNext = i;
		++iNext[side];
		int indexNext = INDEXV(iNext);

		int interfaceIndex = side + DIM * index;
		
		real8 stateL2 = stateBuffer[indexPrev2];
		real8 stateL = stateBuffer[indexPrev];
		real8 stateR = stateBuffer[index];
		real8 stateR2 = stateBuffer[indexNext];

		//q0 = d/dx ln alpha
		//q1 = d/dx ln g = d/dx ln g_xx
		
		real ln_alphaL = (stateR.s[0] - stateL2.s[0]) / (2.f * dx[side]);
		real alphaL = exp(ln_alphaL);
		real ln_gL = (stateR.s[1] - stateL2.s[1]) / (2.f * dx[side]);
		real gL = exp(ln_gL);
		real weightL = .5f;
		
		real ln_alphaR = (stateR2.s[0] - stateL.s[0]) / (2.f * dx[side]);
		real alphaR = exp(ln_alphaR);
		real ln_gR = (stateR2.s[1] - stateL.s[1]) / (2.f * dx[side]);
		real gR = exp(ln_gR);
		real weightR = .5f;
	
		real invDenom = weightL + weightR;
		real alpha = (alphaL * weightL + alphaR * weightR) * invDenom;
		real g = (gL * weightL + gR * weightR) * invDenom;

		const real f = 1.f;	//Bona Masso slicing condition
		
		real sqrtF = sqrt(f);
		real oneOverF = 1.f / f;
		real oneOverSqrtF = sqrt(oneOverF);
		
		real sqrtG = sqrt(g);
		real oneOverSqrtG = 1.f / sqrtG;	
		
		//eigenvalues

		real eigenvalue = alpha * sqrtF * oneOverSqrtG;
		real8 eigenvalues = (real8)(
			-eigenvalue,
			0.f,
			eigenvalue,
			0.f,
			0.f,
			0.f,
			0.f,
			0.f);
		
		eigenvaluesBuffer[interfaceIndex] = eigenvalues;

		//eigenvectors

		//specify transposed
		real16 eigenvectorsA;
		real16 eigenvectorsB;
		real16 eigenvectorsC;
		real16 eigenvectorsD;
		
		//col
		eigenvectorsA.s0 = f;
		eigenvectorsA.s8 = 2.f; 
		eigenvectorsB.s0 = -sqrtF;
		eigenvectorsB.s8 = 0.f;
		eigenvectorsC.s0 = 0.f;
		eigenvectorsC.s8 = 0.f;
		eigenvectorsD.s0 = 0.f;
		eigenvectorsD.s8 = 0.f;
		//col
		eigenvectorsA.s1 = 0.f;
		eigenvectorsA.s9 = 1.f;
		eigenvectorsB.s1 = 0.f;
		eigenvectorsB.s9 = 0.f;
		eigenvectorsC.s1 = 0.f;
		eigenvectorsC.s9 = 0.f;
		eigenvectorsD.s1 = 0.f;
		eigenvectorsD.s9 = 0.f;
		//col
		eigenvectorsA.s2 = f;
		eigenvectorsA.sA = 2.f;
		eigenvectorsB.s2 = sqrtF;
		eigenvectorsB.sA = 0.f;
		eigenvectorsC.s2 = 0.f;
		eigenvectorsC.sA = 0.f;
		eigenvectorsD.s2 = 0.f;
		eigenvectorsD.sA = 0.f;
		//col
		eigenvectorsA.s3 = 0.f;
		eigenvectorsA.sB = 0.f;
		eigenvectorsB.s3 = 0.f;
		eigenvectorsB.sB = 0.f;
		eigenvectorsC.s3 = 0.f;
		eigenvectorsC.sB = 0.f;
		eigenvectorsD.s3 = 0.f;
		eigenvectorsD.sB = 0.f;	
		//col
		eigenvectorsA.s4 = 0.f;
		eigenvectorsA.sC = 0.f;
		eigenvectorsB.s4 = 0.f;
		eigenvectorsB.sC = 0.f;
		eigenvectorsC.s4 = 0.f;
		eigenvectorsC.sC = 0.f;
		eigenvectorsD.s4 = 0.f;
		eigenvectorsD.sC = 0.f;	

		eigenvectorsBuffer[0 + 4 * interfaceIndex] = eigenvectorsA;
		eigenvectorsBuffer[1 + 4 * interfaceIndex] = eigenvectorsB;
		eigenvectorsBuffer[2 + 4 * interfaceIndex] = eigenvectorsC;
		eigenvectorsBuffer[3 + 4 * interfaceIndex] = eigenvectorsD;

		//calculate eigenvector inverses ... 
		eigenvectorsInverseBuffer[0 + 4 * interfaceIndex] = (real16)( 
		//min row
			.5f * oneOverF,
			0.f,
			-.5f * oneOverSqrtF,
			0.f,
			0.f,
			0.f,
			0.f, 
			0.f,
		//mid normal row
			-2.f * oneOverF,
			1.f,
			0.f,
			0.f,
			0.f,
			0.f,
			0.f,
			0.f);
		eigenvectorsInverseBuffer[1 + 4 * interfaceIndex] = (real16)(
		//mid tangent A row
			.5f * oneOverF,
			0.f,
			.5f * oneOverSqrtF,
			0.f,	
			0.f,
			0.f,
			0.f,
			0.f,
		//mid tangent B row
			0.f,
			0.f,
			0.f,
			0.f,
			0.f,
			0.f,
			0.f,
			0.f);
		eigenvectorsInverseBuffer[2 + 4 * interfaceIndex] = (real16)(
		//max row
			0.f,
			0.f,
			0.f,
			0.f,
			0.f,
			0.f,
			0.f,
			0.f,
			
			0.f,
			0.f,
			0.f,
			0.f,
			0.f,
			0.f,
			0.f,
			0.f);
		eigenvectorsInverseBuffer[3 + 4 * interfaceIndex] = (real16)(
			0.f,
			0.f,
			0.f,
			0.f,
			0.f,
			0.f,
			0.f,
			0.f,
			
			0.f,
			0.f,
			0.f,
			0.f,
			0.f,
			0.f,
			0.f,
			0.f);
	}
}

