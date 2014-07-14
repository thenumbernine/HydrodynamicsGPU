#pragma once

#include "HydroGPU/Equation/Equation.h"
#include <vector>
#include <string>

struct Solver;

namespace HydroGPU {
namespace Equation {

struct Euler : public Equation {
	typedef Equation Super;
	Euler(Solver& solver);
	virtual void getProgramSources(Solver& solver, std::vector<std::string>& sources);
	virtual int stateGetBoundaryKernelForBoundaryMethod(Solver& solver, int dim, int state);
	virtual int gravityGetBoundaryKernelForBoundaryMethod(Solver& solver, int dim);
};

}
}

