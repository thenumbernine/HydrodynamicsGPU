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

