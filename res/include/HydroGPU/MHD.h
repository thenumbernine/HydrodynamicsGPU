#pragma once

#include "HydroGPU/Euler.h"

typedef struct {
	real density;
	real pressure;
	real pressureTotal;
	real enthalpyTotal;
	real4 velocity;
	real4 magneticField;
} Primitives_t;

typedef struct {
	real slow;
	real Alfven;
	real fast;
} Wavespeed_t;
