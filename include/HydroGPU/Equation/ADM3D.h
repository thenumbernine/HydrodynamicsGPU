#pragma once

#include "HydroGPU/Equation/Equation.h"
#include <vector>
#include <string>

namespace HydroGPU {
namespace Solver {
struct Solver;
}
namespace Equation {

//most likely pseudo-cartesian coordinates
struct ADM3D : public Equation {
	typedef Equation Super;
	ADM3D(HydroGPU::Solver::Solver* solver);
	virtual void getProgramSources(std::vector<std::string>& sources);
	virtual int stateGetBoundaryKernelForBoundaryMethod(int dim, int state);
	virtual int gravityGetBoundaryKernelForBoundaryMethod(int dim);
};

}
}


