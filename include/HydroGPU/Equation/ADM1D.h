#pragma once

#include "HydroGPU/Equation/Equation.h"
#include <vector>
#include <string>

namespace HydroGPU {
struct HydroGPUApp;
namespace Equation {

/*
This is an implementation of my first working 1D ADM simulation
Someday I'll replace it with the 3-variable system that I have an implementation of in my gravitational-wave-simulation project
that is based on the Alcubierre "Introduction to 3+1 Numerical Relativity" book.

implementing that system would mean integrating sources for 5 variables
but performing roe matrices on 3 variables

two solutions to this:
1) separate alpha and g into a separate texture, add it to the param list of update sources
2) modify underlying roe routines to operate on a subset of states
*/
struct ADM1D : public Equation {
	using Super = Equation;
	ADM1D(HydroGPUApp* app_);
	virtual void getProgramSources(std::vector<std::string>& sources);
	virtual int stateGetBoundaryKernelForBoundaryMethod(int dim, int state, int minmax);
	virtual std::string name() const { return "ADM1D"; }
};

}
}
