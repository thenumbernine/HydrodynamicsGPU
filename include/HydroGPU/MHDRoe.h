#pragma once

#include "HydroGPU/Roe.h"

struct HydroGPUApp;

/*
Roe solver for MHD equations
*/
struct MHDRoe : public Roe {
	typedef Roe Super;
	MHDRoe(HydroGPUApp& app);
protected:
	virtual std::vector<std::string> getProgramSources();
};

