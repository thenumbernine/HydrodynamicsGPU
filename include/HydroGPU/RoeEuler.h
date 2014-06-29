#pragma once

#include "HydroGPU/Roe.h"

struct HydroGPU;

/*
Roe solver for Euler equations
*/
struct RoeEuler : public Roe {
	typedef Roe Super;
	RoeEuler(HydroGPUApp& app);
};

