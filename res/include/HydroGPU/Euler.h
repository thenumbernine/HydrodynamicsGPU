#pragma once

#include "HydroGPU/Shared/Common.h"

//velocity
#if EULER_DIM == 1
#define VELOCITY(ptr)	((real4)((ptr)[STATE_MOMENTUM_X], 0.f, 0.f, 0.f) / (ptr)[STATE_DENSITY])
#elif EULER_DIM == 2
#define VELOCITY(ptr)	((real4)((ptr)[STATE_MOMENTUM_X], (ptr)[STATE_MOMENTUM_Y], 0.f, 0.f) / (ptr)[STATE_DENSITY])
#elif EULER_DIM == 3
#define VELOCITY(ptr)	((real4)((ptr)[STATE_MOMENTUM_X], (ptr)[STATE_MOMENTUM_Y], (ptr)[STATE_MOMENTUM_Z], 0.f) / (ptr)[STATE_DENSITY])
#endif
