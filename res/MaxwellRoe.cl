/*
Hyperbolic formalism of Maxwell's equations
Described in Trangenstein, "Numeric Solutions of Hyperbolic Partial Differential Equations" Section 4.3
*/

#include "HydroGPU/Shared/Common.h"

#define ELECTRIC_FIELD(x) (real4)(x[STATE_ELECTRIC_X], x[STATE_ELECTRIC_Y], x[STATE_ELECTRIC_Z], 0.f)
#define MAGNETIC_FIELD(x) (real4)(x[STATE_MAGNETIC_X], x[STATE_MAGNETIC_Y], x[STATE_MAGNETIC_Z], 0.f)

constant float sqrt_1_2 = 0.7071067811865475727373109293694142252206802368164f;

void leftEigenvectorTransform(
	real* results,
	const __global real* eigenvectorData,	//not used
	const real* input,
	int side)
{
	real4 electric = ELECTRIC_FIELD(input);
	real4 magnetic = MAGNETIC_FIELD(input);

	//swap input dim x<->side
	if (side == 1) {
		electric.xy = electric.yx;
		magnetic.xy = magnetic.yx;
	} else if (side == 2) {
		electric.xz = electric.zx;
		magnetic.xz = magnetic.zx;
	}
	
	const float se = sqrtPermittivity * sqrt_1_2;
	const float su = sqrtPermeability * sqrt_1_2;
	const float ise = 1.f / se;
	const float isu = 1.f / su;

	results[0] = electric.z * ise + magnetic.y * isu;
	results[1] = electric.y * -ise + magnetic.z * isu;
	results[2] = electric.x * -ise + magnetic.x * isu;
	results[3] = electric.x * ise + magnetic.x * isu;
	results[4] = electric.y * ise + magnetic.z * isu;
	results[5] = electric.z * -ise + magnetic.y * isu;

/*
	//swap output dim x<->side
	real tmp;
	if (side == 1) {
		tmp = results[0];
		results[0] = results[1];
		results[1] = tmp;
		tmp = results[3];
		results[3] = results[4];
		results[4] = tmp;
	} else if (side == 2) {
		tmp = results[0];
		results[0] = results[4];
		results[4] = tmp;
		tmp = results[3];
		results[3] = results[5];
		results[5] = tmp;
	}
*/
}

void rightEigenvectorTransform(
	__global real* results,
	const __global real* eigenvector,	//not used
	const real* input,
	int side)
{
	real4 electric = ELECTRIC_FIELD(input);
	real4 magnetic = MAGNETIC_FIELD(input);

/*	//swap input dim x<->side
	if (side == 1) {
		electric.xy = electric.yx;
		magnetic.xy = magnetic.yx;
	} else if (side == 2) {
		electric.xz = electric.zx;
		magnetic.xz = magnetic.zx;
	}
*/	
	const float se = sqrtPermittivity * sqrt_1_2;
	const float su = sqrtPermeability * sqrt_1_2;

	results[0] = electric.z * -se + magnetic.x * se;
	results[1] = electric.y * -se + magnetic.y * se;
	results[2] = electric.x * se + magnetic.z * -se;
	results[3] = electric.z * su + magnetic.x * su;
	results[4] = electric.x * su + magnetic.z * su;
	results[5] = electric.y * su + magnetic.y * su;

	//swap output dim x<->side
	real tmp;
	if (side == 1) {
		tmp = results[0];
		results[0] = results[1];
		results[1] = tmp;
		tmp = results[3];
		results[3] = results[4];
		results[4] = tmp;
	} else if (side == 2) {
		tmp = results[0];
		results[0] = results[4];
		results[4] = tmp;
		tmp = results[3];
		results[3] = results[5];
		results[5] = tmp;
	}
}

__kernel void calcEigenBasisSide(
	__global real* eigenvaluesBuffer,
	__global real* eigenvectorsBuffer,	//not used
	const __global real* stateBuffer,
	int side)
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

	int interfaceIndex = index;
	
	__global real* eigenvalues = eigenvaluesBuffer + NUM_STATES * interfaceIndex;

	//eigenvalues

	real eigenvalue = 1.f / (sqrtPermittivity * sqrtPermeability); 
	eigenvalues[0] = -eigenvalue;
	eigenvalues[1] = -eigenvalue;
	eigenvalues[2] = 0.f;
	eigenvalues[3] = 0.f;
	eigenvalues[4] = eigenvalue;
	eigenvalues[5] = eigenvalue;
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

//for some odd reason, with source I'm getting bias in movement to the left and
return;
//I'm also getting reflections off the right-hand side, regradless of source

	const __global real* state = stateBuffer + NUM_STATES * index;
	__global real* deriv = derivBuffer + NUM_STATES * index;
	
	real4 conductiveElectric = (real4)(state[STATE_ELECTRIC_X], state[STATE_ELECTRIC_Y], state[STATE_ELECTRIC_Z], 0.f) * (conductivity / permittivity);
	
	deriv[STATE_ELECTRIC_X] -= conductiveElectric.x;
	deriv[STATE_ELECTRIC_Y] -= conductiveElectric.y;
	deriv[STATE_ELECTRIC_Z] -= conductiveElectric.z;
}
