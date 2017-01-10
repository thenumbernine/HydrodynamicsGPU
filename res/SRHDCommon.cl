//note that, unlike src/Equation/Euler.cpp, res/EulerMHDCommon.cl, res/EulerRoe.cl, this file has the grid dimension and the equation dimension tied together

#include "HydroGPU/Shared/Common.h"

//velocity
#if DIM == 1
#define VELOCITY(ptr)	((real4)((ptr)[PRIMITIVE_VELOCITY_X], 0., 0., 0.))
#elif DIM == 2
#define VELOCITY(ptr)	((real4)((ptr)[PRIMITIVE_VELOCITY_X], (ptr)[PRIMITIVE_VELOCITY_Y], 0., 0.))
#elif DIM == 3
#define VELOCITY(ptr)	((real4)((ptr)[PRIMITIVE_VELOCITY_X], (ptr)[PRIMITIVE_VELOCITY_Y], (ptr)[PRIMITIVE_VELOCITY_Z], 0.))
#endif

#define gamma idealGas_heatCapacityRatio	//laziness

//#ifdef AMD_SUCKS...
//#pragma OPENCL EXTENSION cl_amd_printf : enable

/*
Incoming is Newtonian Euler equation state variables: density, momentum, newtonian total energy density
Outgoing is the SRHD primitives associated with it: density, velocity?, and pressure
		and the SRHD state variables:
*/
__kernel void initVariables(
	__global real* stateBuffer,
	__global real* primitiveBuffer)
{
	int4 i = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);
	int index = INDEXV(i);

	__global real* state = stateBuffer + NUM_STATES * index;
	__global real* primitive = primitiveBuffer + NUM_STATES * index;
/*
special modification for SRHD
 until I put more thought into the issue of unifying all iniital state variables for all solvers
I could go back to labelling initial states
then change the problems to provide either internal specific energy or newtonian pressure
and change the solvers C++ code to apply these transformations...
or I could just provide separate initial states for all Euler/MDH and all SRHD equations ...
or I could provide a wrapper like this ...

I usually write out variable names in this project,
but for SRHD we have variables like rho = "rest-mass density" and D = "rest-mass density from Eulerian frame"
... sooo ... 
watch me for the changes
and try to keep up
*/
	// calculate newtonian primitives from state vector
	real rho = state[0];
	real4 v = (real4)(0., 0., 0., 0.);
	v.x = state[1] / state[0];
#if DIM > 1
	v.y = state[2] / state[0];
#endif
#if DIM > 2
	v.z = state[3] / state[0];
#endif
	real ETotalClassic = state[DIM+1];
	real vSq = dot(v, v);	
	real eKinClassic = .5 * vSq;
	real eTotalClassic = ETotalClassic / rho;
	real eInt = eTotalClassic - eKinClassic;
	// recast them as SR state variables
	//TODO this whooole process can be implemented in Equation::SRHD::readStateCell
	real P = (gamma - 1.) * rho * eInt;
	real h = 1. + eInt + P / rho; 
	real WSq = 1. / (1. - vSq);
	real W = sqrt(WSq);
	real D = rho * W;
	real4 S = rho * h * WSq * v;
	real tau = rho * h * WSq - P - D;
	//write primitives
	primitive[PRIMITIVE_DENSITY] = rho;
	primitive[PRIMITIVE_VELOCITY_X] = v.x;
#if DIM > 1
	primitive[PRIMITIVE_VELOCITY_Y] = v.y;
#endif
#if DIM > 2
	primitive[PRIMITIVE_VELOCITY_Z] = v.z;
#endif
	primitive[PRIMITIVE_SPECIFIC_INTERNAL_ENERGY] = eInt;
	//write conservatives
	state[STATE_REST_MASS_DENSITY] = D;
	state[STATE_MOMENTUM_DENSITY_X] = S.x;
#if DIM > 1
	state[STATE_MOMENTUM_DENSITY_Y] = S.y;
#endif
#if DIM > 2
	state[STATE_MOMENTUM_DENSITY_Z] = S.z;
#endif
	state[STATE_TOTAL_ENERGY_DENSITY] = tau;
}

__kernel void constrainState(
	__global real* stateBuffer)
{
	int4 i = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);
	int index = INDEXV(i);
	__global real* state = stateBuffer + NUM_STATES * index;
	
	//constraining conservative values directly doesn't seem to be physical
	//seems it's a better idea to constrain fluxes to ensure they don't cause conservative values to be violated
	
	state[STATE_REST_MASS_DENSITY] = max(state[STATE_REST_MASS_DENSITY], srhd_DMin);
	state[STATE_TOTAL_ENERGY_DENSITY] = max(state[STATE_TOTAL_ENERGY_DENSITY], srhd_tauMin);
	
	state[STATE_REST_MASS_DENSITY] = min(state[STATE_REST_MASS_DENSITY], srhd_DMax);
	state[STATE_TOTAL_ENERGY_DENSITY] = min(state[STATE_TOTAL_ENERGY_DENSITY], srhd_tauMax);
}

// convert conservative to primitive using root-finding
//From Marti & Muller 2008
__kernel void updatePrimitives(
	__global real* primitiveBuffer,
	const __global real* stateBuffer)
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

	const __global real* state = stateBuffer + NUM_STATES * index;
	real D = state[STATE_REST_MASS_DENSITY];
	real4 S = (real4)(0., 0., 0., 0.);
	S.x = state[STATE_MOMENTUM_DENSITY_X];
#if DIM > 1
	S.y = state[STATE_MOMENTUM_DENSITY_Y];
#endif
#if DIM > 2
	S.z = state[STATE_MOMENTUM_DENSITY_Z];
#endif
	real tau = state[STATE_TOTAL_ENERGY_DENSITY];

//printf("cell %d cons= %f %f %f\n", index, D, S.x, tau);

	__global real* primitive = primitiveBuffer + NUM_STATES * index;
	//real rho = primitive[PRIMITIVE_DENSITY];
	real4 v = (real4)(0., 0., 0., 0.);
	v.x = primitive[PRIMITIVE_VELOCITY_X];
#if DIM > 1
	v.y = primitive[PRIMITIVE_VELOCITY_Y];
#endif
#if DIM > 2
	v.z = primitive[PRIMITIVE_VELOCITY_Z];
#endif
	//real eInt = primitive[PRIMITIVE_SPECIFIC_INTERNAL_ENERGY];

	real SLen = length(S);
	real PMin = max(SLen - tau - D + SLen * srhd_solvePrimVelEpsilon, srhd_solvePrimPMinEpsilon);
	real PMax = (gamma - 1.) * tau;
	PMax = max(PMax, PMin);
	real P = .5 * (PMin + PMax);

	for (int iter = 0; iter < srhd_solvePrimMaxIter; ++iter) {
		real vLen = SLen / (tau + D + P);
		real vSq = vLen * vLen;
		real W = 1. / sqrt(1. - vSq);
		real eInt = (tau + D * (1. - W) + P * (1. - W*W)) / (D * W);
		real rho = D / W;
		real f = (gamma - 1.) * rho * eInt - P;
		real csSq = (gamma - 1.) * (tau + D * (1. - W) + P) / (tau + D + P);
		real df_dP = vSq * csSq - 1.;
		real newP = P - f / df_dP;
		newP = max(newP, PMin);
		real PError = fabs(1. - newP / P);
		P = newP;
		if (PError < srhd_solvePrimStopEpsilon) {
			v = S / (tau + D + P);
			W = 1. / sqrt(1. - dot(v,v));
			rho = D / W;
			rho = max(rho, srhd_rhoMin);
			rho = min(rho, srhd_rhoMax);
			eInt = P / (rho * (gamma - 1.));
			eInt = min(eInt, srhd_eIntMax);
			primitive[PRIMITIVE_DENSITY] = rho;
			primitive[PRIMITIVE_VELOCITY_X] = v.x;
#if DIM > 1
			primitive[PRIMITIVE_VELOCITY_Y] = v.y;
#endif
#if DIM > 2
			primitive[PRIMITIVE_VELOCITY_Z] = v.z;
#endif
			primitive[PRIMITIVE_SPECIFIC_INTERNAL_ENERGY] = eInt;
//printf("cell %d finished with prims = %f %f %f\n", index, rho, v.x, eInt);
			return;
		}
	}
//printf("cell %d didn't finish\n", index);
}

//specific to Euler equations
__kernel void convertToTex(
#ifdef has_gl_sharing 
	__write_only image3d_t destTex,
#else
	global float4* destTex,
#endif
	int displayMethod,
	const __global real* stateBuffer,
	const __global real* primitiveBuffer)
//const __global real* potentialBuffer		
//TODO get SRHD equation working with selfgrav by renaming STATE_REST_MASS_DENSITY to STATE_DENSITY
{
	int4 i = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);
	int index = INDEXV(i);

	const __global real* state = stateBuffer + NUM_STATES * index;
	const __global real* primitive = primitiveBuffer + NUM_STATES * index;

	real rho = primitive[PRIMITIVE_DENSITY];
	real eInt = primitive[PRIMITIVE_SPECIFIC_INTERNAL_ENERGY];

	real value;
	if (displayMethod == DISPLAY_DENSITY) {
		value = rho;
	} else if (displayMethod == DISPLAY_VELOCITY_X) {
		value = primitive[PRIMITIVE_VELOCITY_X];
#if DIM > 1
	} else if (displayMethod == DISPLAY_VELOCITY_Y) {
		value = primitive[PRIMITIVE_VELOCITY_Y];
#endif
#if DIM > 2
	} else if (displayMethod == DISPLAY_VELOCITY_Z) {
		value = primitive[PRIMITIVE_VELOCITY_Z];
#endif
	} else if (displayMethod == DISPLAY_VELOCITY_MAGN) {
		real vSq = primitive[PRIMITIVE_VELOCITY_X] * primitive[PRIMITIVE_VELOCITY_X];
#if DIM > 1
		vSq += primitive[PRIMITIVE_VELOCITY_Y] * primitive[PRIMITIVE_VELOCITY_Y];
#endif
#if DIM > 2
		vSq += primitive[PRIMITIVE_VELOCITY_Z] * primitive[PRIMITIVE_VELOCITY_Z];
#endif
		value = sqrt(vSq);
	} else if (displayMethod == DISPLAY_E_INTERNAL) {
		value = eInt;
	} else if (displayMethod == DISPLAY_P) {
		value = (gamma - 1.) * rho * eInt;
	} else if (displayMethod == DISPLAY_H) {
		value = 1 + gamma * eInt;
	} else if (displayMethod == DISPLAY_D) {
		value = state[STATE_REST_MASS_DENSITY];
	} else if (displayMethod == DISPLAY_S_X) {
		value = state[STATE_MOMENTUM_DENSITY_X];
#if DIM > 1
	} else if (displayMethod == DISPLAY_S_Y) {
		value = state[STATE_MOMENTUM_DENSITY_Y];
#endif
#if DIM > 2
	} else if (displayMethod == DISPLAY_S_Z) {
		value = state[STATE_MOMENTUM_DENSITY_Z];
#endif
	} else if (displayMethod == DISPLAY_S_MAGN) {
		real SSq = state[STATE_MOMENTUM_DENSITY_X] * state[STATE_MOMENTUM_DENSITY_X];
#if DIM > 1
		SSq += state[STATE_MOMENTUM_DENSITY_Y] * state[STATE_MOMENTUM_DENSITY_Y];
#endif
#if DIM > 2
		SSq += state[STATE_MOMENTUM_DENSITY_Z] * state[STATE_MOMENTUM_DENSITY_Z];
#endif
		value = sqrt(SSq);
	} else if (displayMethod == DISPLAY_TAU) {
		value = state[STATE_TOTAL_ENERGY_DENSITY];
	}

#ifdef has_gl_sharing 
	write_imagef(destTex, (int4)(i.x, i.y, i.z, 0), (float4)(value, 0., 0., 0.));
#else
	destTex[index] = (float4)(value, 0., 0., 0.);
#endif
}

constant float2 offset[6] = {
	(float2)(-.5, 0.),
	(float2)(.5, 0.),
	(float2)(.2, .3),
	(float2)(.5, 0.),
	(float2)(.2, -.3),
	(float2)(.5, 0.),
};

__kernel void updateVectorField(
	__global float* vectorFieldVertexBuffer,
	real scale,
	int displayMethod,
	const __global real* stateBuffer,
	const __global real* primitiveBuffer)
{
	int4 i = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);
	int4 size = (int4)(get_global_size(0), get_global_size(1), get_global_size(2), 0);	
	int vertexIndex = i.x + size.x * (i.y + size.y * i.z);
	__global float* vertex = vectorFieldVertexBuffer + 6 * 3 * vertexIndex;
	
	float4 f = (float4)(
		((float)i.x + .5) / (float)size.x,
		((float)i.y + .5) / (float)size.y,
		((float)i.z + .5) / (float)size.z,
		0.);

	//times grid size divided by field size
	float4 sf = (float4)(f.x * SIZE_X, f.y * SIZE_Y, f.z * SIZE_Z, 0.);
	int4 si = (int4)(sf.x, sf.y, sf.z, 0);
	//float4 fp = (float4)(sf.x - (float)si.x, sf.y - (float)si.y, sf.z - (float)si.z, 0.);
	
	int stateIndex = INDEXV(si);
	const __global real* state = stateBuffer + NUM_STATES * stateIndex;
	const __global real* primitive = primitiveBuffer + NUM_STATES * stateIndex;
	
	real4 field = (real4)(0., 0., 0., 0.);
	if (displayMethod == VECTORFIELD_VELOCITY) {
		field.x = primitive[PRIMITIVE_VELOCITY_X];
#if DIM > 1
		field.y = primitive[PRIMITIVE_VELOCITY_Y];
#endif
#if DIM > 2
		field.z = primitive[PRIMITIVE_VELOCITY_Z];
#endif
	} else if (displayMethod == VECTORFIELD_MOMENTUM) {
		field.x = state[STATE_MOMENTUM_DENSITY_X];
#if DIM > 1
		field.y = state[STATE_MOMENTUM_DENSITY_Y];
#endif
#if DIM > 2
		field.z = state[STATE_MOMENTUM_DENSITY_Z];
#endif
	}

	//field is the first axis of the basis to draw the arrows
	//the second should be perpendicular to field
#if DIM < 3
	real4 tv = (real4)(-field.y, field.x, 0., 0.);
#elif DIM == 3
	real4 vx = (real4)(0., -field.z, field.y, 0.);
	real4 vy = (real4)(field.z, 0., -field.x, 0.);
	real4 vz = (real4)(-field.y, field.x, 0., 0.);
	real lxsq = dot(vx,vx);
	real lysq = dot(vy,vy);
	real lzsq = dot(vz,vz);
	real4 tv;
	if (lxsq > lysq) {	//x > y
		if (lxsq > lzsq) {	//x > z, x > y
			tv = vx;
		} else {	//z > x > y
			tv = vz;
		}
	} else {	//y >= x
		if (lysq > lzsq) {	//y >= x, y > z
			tv = vy;
		} else {	// z > y >= x
			tv = vz;
		}
	}
#endif

	for (int i = 0; i < 6; ++i) {
		vertex[0 + 3 * i] = f.x * (XMAX - XMIN) + XMIN + scale * (offset[i].x * field.x + offset[i].y * tv.x);
		vertex[1 + 3 * i] = f.y * (YMAX - YMIN) + YMIN + scale * (offset[i].x * field.y + offset[i].y * tv.y);
		vertex[2 + 3 * i] = f.z * (ZMAX - ZMIN) + ZMIN + scale * (offset[i].x * field.z + offset[i].y * tv.z);
	}
}
