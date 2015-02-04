#pragma once

#include "HydroGPU/Solver/Roe.h"

namespace HydroGPU {
namespace Solver {

/*
Roe solver for Maxwell equations
*/
struct MaxwellRoe : public Roe {
	typedef Roe Super;
	
	cl::Kernel addSourceKernel;
	
	MaxwellRoe(HydroGPUApp& app);
	virtual void init();
protected:
	virtual std::vector<std::string> getProgramSources();
	virtual void calcDeriv(cl::Buffer derivBuffer);
};

}
}


