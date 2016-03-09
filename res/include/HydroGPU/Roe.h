#pragma once

#include "HydroGPU/Shared/Common.h"

void leftEigenvectorTransform(
	real* results,
	const __global real* eigenvectorData,
	const real* input,
	int side);

void rightEigenvectorTransform(
	__global real* results,
	const __global real* eigenvectorData,
	const real* input,
	int side);
