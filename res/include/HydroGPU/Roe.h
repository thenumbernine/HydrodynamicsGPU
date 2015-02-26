#pragma once

#include "HydroGPU/Shared/Common.h"

void eigenfieldTransform(
	real* results,
	const __global real* eigenfield,
	const real* input);

void eigenfieldInverseTransform(
	__global real* results,
	const __global real* eigenfield,
	const real* input);

