#pragma once

#include "HydroGPU/Roe.h"

struct HydroGPU;

/*
Roe solver for Euler equations
*/
struct EulerRoe : public Roe {
	typedef Roe Super;
	using Super::Super;
protected:
	virtual std::vector<std::string> getProgramSources();
};

