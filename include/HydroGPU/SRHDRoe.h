#pragma once

#include "HydroGPU/Roe.h"

struct HydroGPU;

/*
Roe solver for Euler equations
*/
struct SRHDRoe : public Roe {
	typedef Roe Super;
	SRHDRoe(HydroGPUApp& app);
protected:
	virtual std::vector<std::string> getProgramSources();
};

