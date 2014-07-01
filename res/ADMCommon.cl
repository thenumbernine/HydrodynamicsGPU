#include "HydroGPU/Shared/Common.h"

//specific to Euler equations
__kernel void convertToTex(
	const __global real* stateBuffer,
	const __global real* gravityPotentialBuffer,
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

	real ln_alpha = (stateR[STATE_DX_LN_ALPHA] - stateL[STATE_DX_LN_ALPHA]) / (2.f * dx[side]);
	real alpha = exp(ln_alpha);
	real ln_g = (stateR[STATE_DX_LN_G] - stateL[STATE_DX_LN_G]) / (2.f * dx[side]);
	real g = exp(ln_g);
	real KTilde = state[STATE_KTILDE];
	real K = KTilde / sqrt(g);

	float4 color = (float4)(alpha, g, K, 0.f) * displayScale;
	real value;
	switch (displayMethod) {
	case DISPLAY_DENSITY:	//density
		value = density;
		break;
	case DISPLAY_VELOCITY:	//velocity
		value = velocity;
		break;
	case DISPLAY_PRESSURE:	//pressure
		value = (GAMMA - 1.f) * specificEnergyInternal * density;
		break;
	case DISPLAY_GRAVITY_POTENTIAL:
		value = gravityPotentialBuffer[index];
		break;
	default:
		value = .5f;
		break;
	}
	value *= displayScale;

	float4 color = read_imagef(gradientTex, CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_REPEAT | CLK_FILTER_LINEAR, value).bgra;
	write_imagef(fluidTex, (int4)(i.x, i.y, i.z, 0), color);
}

