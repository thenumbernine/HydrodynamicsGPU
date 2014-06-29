#pragma once

#include "HydroGPU/Roe.h"

struct HydroGPU;

/*
Roe solver for MHD equations
*/
struct RoeMHD : public Roe {
	typedef Roe Super;
	RoeMHD(HydroGPUApp& app);
};

