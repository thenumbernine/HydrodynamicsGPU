#pragma once

#include "HydroGPU/Equation/Equation.h"
#include "HydroGPU/Equation/SelfGravitationBehavior.h"
#include <vector>
#include <string>

namespace HydroGPU {
namespace Solver {
struct Solver;
}
namespace Equation {

struct Euler : public SelfGravitationBehavior<Equation> {
	typedef SelfGravitationBehavior<Equation> Super;
	Euler(HydroGPU::Solver::Solver* solver);
	virtual void getProgramSources(std::vector<std::string>& sources);
	virtual int stateGetBoundaryKernelForBoundaryMethod(int dim, int state);
	virtual void readStateCell(real* state, const real* source);
	virtual int numReadStateChannels();
};

}
}

