#pragma once

#include "HydroGPU/Equation/Equation.h"
#include <vector>
#include <string>

namespace HydroGPU {
struct HydroGPUApp;
namespace Equation {

struct Maxwell : public Equation {
	using Super = Equation;
	Maxwell(HydroGPUApp* app_);
	virtual void getProgramSources(std::vector<std::string>& sources);
	virtual int stateGetBoundaryKernelForBoundaryMethod(int dim, int state, int minmax);
	virtual std::string name() const { return "Maxwell"; } 
};

}
}
