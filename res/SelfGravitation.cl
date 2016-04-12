#include "HydroGPU/Shared/Common.h"

/*
Phi = - G m / r
nabla^2 phi = -4 pi G rho

k = 1/dx^2 + 1/dy^2 + 1/dz^2
[-2k 1/dx^2 ... 1/dy^2 ... 1/dz^2           ] [   ]           [   ]
[1/dx^2 -2k 1/dx^2 ... 1/dy^2 ... 1/dz^2    ] [phi] = -4 pi G [rho]
[   1/dx^2 -2k 1/dx^2 ... 1/dy^2 ... 1/dz^2 ] [   ]           [   ]

boundary:
phi(dOmega) = 0

jacobi:

phi_i := (b_i - sum_j (a_ij phi_j) / a_ii), j != i

*/
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

	//sum of skew (non-diag) components: sum_j a_ij phi_j, j != i
	real skewSum = 0.;
	if (i.x < SIZE_X-3) skewSum += gravityPotentialBuffer[index + stepsize.x] / (DX * DX);
	if (i.x > 2) skewSum += gravityPotentialBuffer[index - stepsize.x] / (DX * DX);
#if DIM > 1
	if (i.y < SIZE_Y-3) skewSum += gravityPotentialBuffer[index + stepsize.y] / (DY * DY);
	if (i.y > 2) skewSum += gravityPotentialBuffer[index - stepsize.y] / (DY * DY);
#endif
#if DIM > 2
	if (i.z < SIZE_Z-3) skewSum += gravityPotentialBuffer[index + stepsize.z] / (DZ * DZ);
	if (i.z > 2) skewSum += gravityPotentialBuffer[index - stepsize.z] / (DZ * DZ);
#endif

	const real diag = -2. * (1. / (DX * DX)
#if DIM > 1
							+ 1. / (DY * DY)
#endif
#if DIM > 2
							+ 1. / (DZ * DZ)
#endif
	);

	const real G = selfGrav_gravitationalConstant;		//6.67384e-11 m^3 / (kg s^2)
	real density = stateBuffer[STATE_DENSITY + NUM_STATES * index];
	gravityPotentialBuffer[index] = (4. * M_PI * G * density - skewSum) / diag;
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
	real derivEnergyTotal = 0.;
	for (int side = 0; side < DIM; ++side) {
		int indexPrev = index - stepsize[side];
		int indexNext = index + stepsize[side];
	
		real gradient = (gravityPotentialBuffer[indexNext] - gravityPotentialBuffer[indexPrev]) / (2. * dx[side]);
		real gravity = -gradient;

		//gravitational force = -gradient of gravitational potential
		deriv[side + STATE_MOMENTUM_X] -= density * gravity;
		derivEnergyTotal -= density * gravity * state[side + STATE_MOMENTUM_X];
	}

	deriv[STATE_ENERGY_TOTAL] += derivEnergyTotal;
}
