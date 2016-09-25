#include "HydroGPU/Shared/Common.h"

//specific to Euler equations
__kernel void convertToTex(
	__write_only image3d_t destTex,
	int displayMethod,
	const __global real* stateBuffer)
{
	int4 i = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);
	int index = INDEXV(i);

	int4 iPrev = i;
	iPrev.x = max(0, iPrev.x - 1);
	int indexPrev = INDEXV(iPrev);

	int4 iNext = i;
	iNext.x = min(SIZE_X - 1, iNext.x + 1);
	int indexNext = INDEXV(iNext);

	const __global real* stateL = stateBuffer + NUM_STATES * indexPrev;
	const __global real* state = stateBuffer + NUM_STATES * index;
	const __global real* stateR = stateBuffer + NUM_STATES * indexNext;

	const int side = 0;

	//partial_i ln alpha = a_i
	//partial_j x partial_i ln alpha = 0
	// so find the curl-free component of partial_i ln alpha ... this is an inverse discrete laplacian, which is a big inverse linear operation
	//... or I could do a cheap trick ...
	real ln_alpha = (stateR[STATE_A_X] - stateL[STATE_A_X]) / (2.f * dx[side]);
	real phi = (stateR[STATE_PHI_X] - stateL[STATE_PHI_X]) / (2.f * dx[side]);
#if DIM > 1
	ln_alpha += (stateR[STATE_A_Y] - stateL[STATE_A_Y]) / (2.f * dx[side]);
	phi += (stateR[STATE_PHI_Y] - stateL[STATE_PHI_Y]) / (2.f * dx[side]);
#endif
#if DIM > 2
	ln_alpha += (stateR[STATE_A_Z] - stateL[STATE_A_Z]) / (2.f * dx[side]);
	phi += (stateR[STATE_PHI_Z] - stateL[STATE_PHI_Z]) / (2.f * dx[side]);
#endif
	ln_alpha /= (float)DIM;
	real alpha = exp(ln_alpha);
	phi /= (float)DIM;	
	
	real K = state[STATE_K];

#if DIM == 1
	float4 color = (float4)(alpha, phi, K, 1.f);
#else
	real value;
	switch (displayMethod) {
	case DISPLAY_ALPHA:		//lapse
		value = alpha;
		break;
	case DISPLAY_PHI:		//conformal factor of the hypersurface metric tensor
		value = phi;
		break;
	case DISPLAY_K:			//trace of extrinsic curvature
		value = K;
		break;
	default:
		value = .5f;
		break;
	}

#endif
	write_imagef(destTex, (int4)(i.x, i.y, i.z, 0), (float4)(value, 0.f, 0.f, 0.f));
}

