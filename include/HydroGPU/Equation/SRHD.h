#pragma once

#include "HydroGPU/Equation/Equation.h"
#include <vector>
#include <string>

namespace HydroGPU {
namespace Solver {
struct Solver;
}
namespace Equation {

struct SRHD : public Equation {
	typedef Equation Super;
	SRHD(HydroGPU::Solver::Solver& solver);
	virtual void getProgramSources(HydroGPU::Solver::Solver& solver, std::vector<std::string>& sources);
	virtual int stateGetBoundaryKernelForBoundaryMethod(HydroGPU::Solver::Solver& solver, int dim, int state);
	virtual int gravityGetBoundaryKernelForBoundaryMethod(HydroGPU::Solver::Solver& solver, int dim);
};

}
}

