#pragma once

//Bona-Masso hyperbolic formalism of the ADM equations

#include "HydroGPU/Equation/Equation.h"
#include <vector>
#include <string>

namespace HydroGPU {
struct HydroGPUApp;
namespace Equation {

//most likely pseudo-cartesian coordinates
struct ADM3D : public Equation {
	using Super = Equation;
	ADM3D(HydroGPUApp* app_);
	virtual void getProgramSources(std::vector<std::string>& sources);
	virtual int stateGetBoundaryKernelForBoundaryMethod(int dim, int state, int minmax);
	virtual std::string name() const { return "ADM3D"; }
};

}
}
