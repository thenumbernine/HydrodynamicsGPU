#pragma once

#include "HydroGPU/Solver/Roe.h"

namespace HydroGPU {
namespace Solver {

/*
Roe solver for ADM equations
*/
struct ADMRoe : public Roe {
	typedef Roe Super;
	
	cl::Kernel addSourceKernel;
	
	ADMRoe(HydroGPUApp& app);
	virtual void init();
protected:
	virtual std::vector<std::string> getProgramSources();
	virtual void calcDeriv(cl::Buffer derivBuffer);
};

}
}

