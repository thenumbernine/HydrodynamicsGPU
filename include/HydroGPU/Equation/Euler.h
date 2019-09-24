#pragma once

#include "HydroGPU/Equation/Equation.h"
#include "HydroGPU/Equation/SelfGravitationBehavior.h"
#include <vector>
#include <string>

namespace HydroGPU {
struct HydroGPUApp;
namespace Equation {

struct Euler : public SelfGravitationBehavior<Equation> {
	using Super = SelfGravitationBehavior<Equation>;
	
protected:
	int dim;

public:
	Euler(HydroGPUApp* app_, int dim_ = -1);
	virtual void getProgramSources(std::vector<std::string>& sources);
	virtual int stateGetBoundaryKernelForBoundaryMethod(int dim, int state, int minmax);
	virtual void readStateCell(real* state, const real* source);
	virtual int numReadStateChannels();
	virtual std::string name() const { return "Euler"; } 

	virtual void setupConvertToTexKernelArgs(cl::Kernel convertToTexKernel, Solver::Solver* solver);
	virtual void setupUpdateVectorFieldKernelArgs(cl::Kernel updateVectorFieldKernel, Solver::Solver* solver);
};

}
}
