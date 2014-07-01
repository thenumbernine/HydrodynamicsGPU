#include "HydroGPU/Shared/Common.h"

//velocity
#if DIM == 1
#define VELOCITY(ptr)	((real4)((ptr)[STATE_VELOCITY_X], 0.f, 0.f, 0.f) / (ptr)[STATE_DENSITY])
#elif DIM == 2
#define VELOCITY(ptr)	((real4)((ptr)[STATE_VELOCITY_X], (ptr)[STATE_VELOCITY_Y], 0.f, 0.f) / (ptr)[STATE_DENSITY])
#elif DIM == 3
#define VELOCITY(ptr)	((real4)((ptr)[STATE_VELOCITY_X], (ptr)[STATE_VELOCITY_Y], (ptr)[STATE_VELOCITY_Z], 0.f) / (ptr)[STATE_DENSITY])
#endif

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

	const __global real* state = stateBuffer + NUM_STATES * index;

	real density = state[STATE_DENSITY];
	real energyTotal = state[STATE_ENERGY_TOTAL];
	real velocitySq = state[STATE_VELOCITY_X] * state[STATE_VELOCITY_X];
#if DIM > 1
	velocitySq += state[STATE_VELOCITY_Y] * state[STATE_VELOCITY_Y];
#endif
#if DIM > 2
	velocitySq += state[STATE_VELOCITY_Z] * state[STATE_VELOCITY_Z];
#endif
	velocitySq /= density * density;
	real velocity = sqrt(velocitySq);
	real specificEnergyTotal = energyTotal / density;
	real specificEnergyKinetic = .5f * velocitySq;
	real specificEnergyPotential = gravityPotentialBuffer[index];
	real specificEnergyInternal = specificEnergyTotal - specificEnergyKinetic - specificEnergyPotential;

#if DIM == 1
#if NUM_STATES == 8	//MHD
	real4 magneticField = (real4)(state[STATE_MAGNETIC_FIELD_X], state[STATE_MAGNETIC_FIELD_Y], state[STATE_MAGNETIC_FIELD_Z], 0.f);
	real magneticFieldMagn = length(magneticField);
#else
	real magneticFieldMagn = 0.f;
#endif
	float4 color = (float4)(density, velocity, specificEnergyInternal, magneticFieldMagn) * displayScale;
#else
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
#endif
	write_imagef(fluidTex, (int4)(i.x, i.y, i.z, 0), color);
}

__kernel void poissonRelax(
	__global real* gravityPotentialBuffer,
	const __global real* stateBuffer,
	int4 repeat)
{
	int4 size = (int4)(SIZE_X, SIZE_Y, SIZE_Z, 0);
	int4 i = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);
	int index = INDEXV(i);

	real sum = 0.f;
	for (int side = 0; side < DIM; ++side) {
		int4 iprev = i;
		int4 inext = i;
		if (repeat[side]) {
			iprev[side] = (iprev[side] + size[side] - 1) % size[side];
			inext[side] = (inext[side] + 1) % size[side];
		} else {
			iprev[side] = max(iprev[side] - 1, 0);
			inext[side] = min(inext[side] + 1, size[side] - 1);
		}
		int indexPrev = INDEXV(iprev);
		int indexNext = INDEXV(inext);
		sum += gravityPotentialBuffer[indexPrev] + gravityPotentialBuffer[indexNext];
	}
	
#define M_PI 3.141592653589793115997963468544185161590576171875f
#define GRAVITY_CONSTANT 1.f		//6.67384e-11 m^3 / (kg s^2)
	real scale = M_PI * GRAVITY_CONSTANT * DX;
#if DIM > 1
	scale *= DY; 
#endif
#if DIM > 2
	scale *= DZ; 
#endif
	real density = stateBuffer[STATE_DENSITY + NUM_STATES * index];
	gravityPotentialBuffer[index] = sum / (2.f * (float)DIM) + scale * density;
}

__kernel void addGravity(
	__global real* stateBuffer,
	const __global real* gravityPotentialBuffer,
	const __global real* dtBuffer)
{
	real dt = dtBuffer[0];
	real4 dt_dx = dt / dx;

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

	real density = stateBuffer[STATE_DENSITY + NUM_STATES * index];

	for (int side = 0; side < DIM; ++side) {
		int indexL = index - stepsize[side];
		int indexR = index + stepsize[side];
	
		real gravityGrad = .5f * (gravityPotentialBuffer[indexR] - gravityPotentialBuffer[indexL]);
		
		stateBuffer[side+1 + NUM_STATES * index] -= dt_dx[side] * density * gravityGrad;
		stateBuffer[STATE_ENERGY_TOTAL + NUM_STATES * index] -= dt * density * gravityGrad * stateBuffer[side+1 + NUM_STATES * index];
	}
}

