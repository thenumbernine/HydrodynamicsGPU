#pragma once

#include "HydroGPU/Equation/Equation.h"
#include <vector>
#include <string>

namespace HydroGPU {
namespace Solver {
struct Solver;
}
namespace Equation {

struct Euler : public Equation {
	typedef Equation Super;
	Euler(HydroGPU::Solver::Solver* solver);
	virtual void getProgramSources(std::vector<std::string>& sources);
	virtual int stateGetBoundaryKernelForBoundaryMethod(int dim, int state);
	virtual int gravityGetBoundaryKernelForBoundaryMethod(int dim);
	virtual void readStateCell(real* state, const real* source);
};

}
}

