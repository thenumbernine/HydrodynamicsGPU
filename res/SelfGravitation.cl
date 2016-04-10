#include "HydroGPU/Shared/Common.h"

__kernel void gravityPotentialPoissonRelax(
	__global real* gravityPotentialBuffer,
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

	real sum = (gravityPotentialBuffer[index + stepsize.x] + gravityPotentialBuffer[index - stepsize.x]) / (DX * DX);
#if DIM > 1
	sum += (gravityPotentialBuffer[index + stepsize.y] + gravityPotentialBuffer[index - stepsize.y]) / (DY * DY);
#if DIM > 2
	sum += (gravityPotentialBuffer[index + stepsize.z] + gravityPotentialBuffer[index - stepsize.z]) / (DZ * DZ);
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

	//delta^2 Phi = 4 pi G rho
	const real pi = 3.141592653589793115997963468544185161590576171875f;
	const real G = selfGrav_gravitationalConstant;		//6.67384e-11 m^3 / (kg s^2)
	const real fourPiGRho = -4.f * pi * G;
	real density = stateBuffer[STATE_DENSITY + NUM_STATES * index];
	gravityPotentialBuffer[index] = (fourPiGRho * density - sum) / denom;
}

__kernel void calcGravityDeriv(
	__global real* derivBuffer,
	const __global real* stateBuffer,
	const __global real* gravityPotentialBuffer)
{
	int4 i = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);
	int index = INDEXV(i);
	__global real* deriv = derivBuffer + NUM_STATES * index;
	
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

	const __global real* state = stateBuffer + NUM_STATES * index;

	real density = state[STATE_DENSITY];
	real derivEnergyTotal = 0.f;
	for (int side = 0; side < DIM; ++side) {
		int indexPrev = index - stepsize[side];
		int indexNext = index + stepsize[side];
	
		real gravityPotentialGradient = (gravityPotentialBuffer[indexNext] - gravityPotentialBuffer[indexPrev]) / (2.f * dx[side]);
	
		//gravitational force = -gradient of gravitational potential
		deriv[side + STATE_MOMENTUM_X] -= density * gravityPotentialGradient;
		derivEnergyTotal -= density * gravityPotentialGradient * state[side + STATE_MOMENTUM_X];
	}

	deriv[STATE_ENERGY_TOTAL] += derivEnergyTotal;
}

