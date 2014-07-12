#include "HydroGPU/Shared/Common.h"

__kernel void initVariables(
	__global real* stateBuffer)
{
	int4 i = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);
	int index = INDEXV(i);
	__global real* state = stateBuffer + NUM_STATES * index;
	//comes in rho, mx, my, mz, eTotal, bx, by, bz
	real totalSpecificEnergy = state[4];	//eKinetic + eInternal ... and maybe ePotential?
	real4 magneticField = (real4)(state[5], state[6], state[7], 0.f);
	real magneticFieldSq = dot(magneticField, magneticField);
	//goes out
	state[STATE_MAGNETIC_FIELD_X] = magneticField.x;
	state[STATE_MAGNETIC_FIELD_Y] = magneticField.y;
	state[STATE_MAGNETIC_FIELD_Z] = magneticField.z;
	state[STATE_ENERGY_TOTAL] = totalSpecificEnergy + .5f * magneticFieldSq / MU0;
}

