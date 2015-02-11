#include "HydroGPU/Shared/Common.h"

__kernel void calcMagneticFieldDivergence(
	__global real* magneticFieldDivergenceBuffer,
	const __global real* stateBuffer)
{
	int4 i = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);
	if (i.x < 2 || i.x >= SIZE_X - 2 
#if DIM > 1
		|| i.y < 2 || i.y >= SIZE_Y - 2 
#if DIM > 2
		|| i.z < 2 || i.z >= SIZE_Z - 2
#endif
#endif
	) {
		return;
	}
	int index = INDEXV(i);

	real divergence = (stateBuffer[STATE_MAGNETIC_FIELD_X + NUM_STATES * (index + stepsize.x)]
		- stateBuffer[STATE_MAGNETIC_FIELD_X + NUM_STATES * (index - stepsize.x)]) / (2.f * DX);
#if DIM > 1
	divergence += (stateBuffer[STATE_MAGNETIC_FIELD_Y + NUM_STATES * (index + stepsize.y)]
		- stateBuffer[STATE_MAGNETIC_FIELD_Y + NUM_STATES * (index - stepsize.y)]) / (2.f * DY);
#if DIM > 2
	divergence += (stateBuffer[STATE_MAGNETIC_FIELD_Z + NUM_STATES * (index + stepsize.z)]
		- stateBuffer[STATE_MAGNETIC_FIELD_Z + NUM_STATES * (index - stepsize.z)]) / (2.f * DZ);
#endif
#endif

	magneticFieldDivergenceBuffer[index] = divergence;
}

__kernel void magneticPotentialPoissonRelax(
	__global real* magneticFieldPotentialBuffer,
	const __global real* magneticFieldDivergenceBuffer)
{
	int4 i = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);
	if (i.x < 2 || i.x >= SIZE_X - 2 
#if DIM > 1
		|| i.y < 2 || i.y >= SIZE_Y - 2 
#if DIM > 2
		|| i.z < 2 || i.z >= SIZE_Z - 2
#endif
#endif
	) {
		return;
	}
	int index = INDEXV(i);

	real sum = (magneticFieldPotentialBuffer[index + stepsize.x] + magneticFieldPotentialBuffer[index - stepsize.x]) / (DX * DX);
#if DIM > 1
	sum += (magneticFieldPotentialBuffer[index + stepsize.y] + magneticFieldPotentialBuffer[index - stepsize.y]) / (DY * DY);
#if DIM > 2
	sum += (magneticFieldPotentialBuffer[index + stepsize.z] + magneticFieldPotentialBuffer[index - stepsize.z]) / (DZ * DZ);
#endif
#endif

	const real denom = -2.f * (1.f / (DX * DX)
#if DIM > 1
		+ 1.f / (DY * DY)
#endif
#if DIM > 2
		+ 1.f / (DZ * DZ)
#endif
	);

	//TODO double-buffer?
	magneticFieldPotentialBuffer[index] = (magneticFieldDivergenceBuffer[index] - sum) / denom;
}

__kernel void magneticFieldRemoveDivergence(
	__global real* stateBuffer,
	const __global real* magneticFieldPotentialBuffer)
{
	int4 i = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);
	if (i.x < 2 || i.x >= SIZE_X - 2 
#if DIM > 1
		|| i.y < 2 || i.y >= SIZE_Y - 2 
#if DIM > 2
		|| i.z < 2 || i.z >= SIZE_Z - 2
#endif
#endif
	) {
		return;
	}
	int index = INDEXV(i);

	stateBuffer[STATE_MAGNETIC_FIELD_X + NUM_STATES * index] -= (magneticFieldPotentialBuffer[index + stepsize.x] - magneticFieldPotentialBuffer[index - stepsize.x]) / (2.f * DX);
#if DIM > 1
	stateBuffer[STATE_MAGNETIC_FIELD_Y + NUM_STATES * index] -= (magneticFieldPotentialBuffer[index + stepsize.y] - magneticFieldPotentialBuffer[index - stepsize.y]) / (2.f * DY);
#if DIM > 2
	stateBuffer[STATE_MAGNETIC_FIELD_Z + NUM_STATES * index] -= (magneticFieldPotentialBuffer[index + stepsize.z] - magneticFieldPotentialBuffer[index - stepsize.z]) / (2.f * DZ);
#endif
#endif
}

