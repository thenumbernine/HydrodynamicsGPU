#pragma once

#include "HydroGPU/Roe.h"

struct HydroGPU;

/*
Roe solver for ADM equations
*/
struct ADMRoe : public Roe {
	typedef Roe Super;
	using Super::Super;
protected:
	virtual std::vector<std::string> getProgramSources();
};

