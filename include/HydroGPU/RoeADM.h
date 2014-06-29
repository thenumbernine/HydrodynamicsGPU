#pragma once

#include "HydroGPU/Roe.h"

struct HydroGPU;

/*
Roe solver for ADM equations
*/
struct RoeADM : public Roe {
	typedef Roe Super;
	RoeADM(HydroGPUApp& app);
};

