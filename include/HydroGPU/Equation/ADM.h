#pragma once

#include "HydroGPU/Equation/Equation.h"
#include <vector>
#include <string>

namespace HydroGPU {
namespace Solver {
struct Solver;
}
namespace Equation {

/*
This is an implementation of my first working 1D ADM simulation
Someday I'll replace it with the 3-variable system that I have an implementation of in my gravitational-wave-simulation project
hat is based on the Alcubierre "Introduction to 3+1 Numerical Relativity" book.
*/
struct ADM : public Equation {
	typedef Equation Super;
	ADM(HydroGPU::Solver::Solver* solver);
	virtual void getProgramSources(std::vector<std::string>& sources);
	virtual int stateGetBoundaryKernelForBoundaryMethod(int dim, int state);
	virtual int gravityGetBoundaryKernelForBoundaryMethod(int dim);
};

}
}

