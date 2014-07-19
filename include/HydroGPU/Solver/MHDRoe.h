#pragma once

#include "HydroGPU/Solver/Roe.h"

namespace HydroGPU {
struct HydroGPUApp;
namespace Solver {

/*
Roe solver for MHD equations
*/
struct MHDRoe : public Roe {
	typedef Roe Super;
	MHDRoe(HydroGPUApp& app);
protected:
	cl::Kernel initVariablesKernel;
	cl::Kernel addMHDSourceKernel;
	
	virtual void initKernels();
	virtual std::vector<std::string> getProgramSources();
	virtual void resetState();

	virtual void calcDeriv(cl::Buffer derivBuffer);
};

}
}

