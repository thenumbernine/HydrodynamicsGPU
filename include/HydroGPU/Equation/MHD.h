#pragma once

#include "HydroGPU/Equation/Euler.h"

namespace HydroGPU {
struct HydroGPUApp;
namespace Equation {

struct MHD : public Euler {
	using Super = Euler;
	MHD(HydroGPUApp* app_);
	virtual std::string name() const { return "MHD"; } 
	virtual void getProgramSources(std::vector<std::string>& sources);
	virtual int stateGetBoundaryKernelForBoundaryMethod(int dim, int state, int minmax);
	virtual void setupConvertToTexKernelArgs(cl::Kernel convertToTexKernel, Solver::Solver* solver);
};

}
}
