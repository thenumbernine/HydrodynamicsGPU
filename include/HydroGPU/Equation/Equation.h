#pragma once

#include "HydroGPU/Shared/Common.h"	//cl shared header
#include <vector>
#include <string>

namespace HydroGPU {
namespace Solver {
struct Solver;
}
namespace Equation {

struct Equation {
protected:
	HydroGPU::Solver::Solver* solver;
public:
	std::vector<std::string> displayMethods;
	std::vector<std::string> boundaryMethods;
	std::vector<std::string> states;

	Equation(HydroGPU::Solver::Solver* solver_);	
	virtual void getProgramSources(std::vector<std::string>& sources);
	virtual int stateGetBoundaryKernelForBoundaryMethod(int dim, int state) = 0;
	std::string buildEnumCode(const std::string& prefix, const std::vector<std::string>& enumStrs);
	virtual void readStateCell(real* state, const real* source);
	virtual int numReadStateChannels();
};

}
}

