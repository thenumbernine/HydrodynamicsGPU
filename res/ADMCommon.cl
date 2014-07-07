#include "HydroGPU/Shared/Common.h"

//specific to Euler equations
__kernel void convertToTex(
	const __global real* stateBuffer,
	const __global real* potentialBuffer,
	__write_only image3d_t fluidTex,
	__read_only image1d_t gradientTex,
	int displayMethod,
	float displayScale)
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

	real ln_alpha = (stateR[STATE_DX_LN_ALPHA] - stateL[STATE_DX_LN_ALPHA]) / (2.f * dx[side]);
	real alpha = exp(ln_alpha);
	real ln_g = (stateR[STATE_DX_LN_G] - stateL[STATE_DX_LN_G]) / (2.f * dx[side]);
	real g = exp(ln_g);
	real KTilde = state[STATE_K_TILDE];
	real K = KTilde / sqrt(g);
	
	float4 color = (float4)(alpha, g, K, 0.f) * displayScale;
	write_imagef(fluidTex, (int4)(i.x, i.y, i.z, 0), color);
}

//no support for this in ADMEquation
//TODO shouldn't even be linking it

__kernel void poissonRelax(
	__global real* potentialBuffer,
	const __global real* stateBuffer,
	int4 repeat)
{
}

__kernel void addGravity(
	__global real* stateBuffer,
	const __global real* potentialBuffer,
	const __global real* dtBuffer)
{
}

