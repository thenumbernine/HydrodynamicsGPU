#pragma once

#include "HydroGPU/Shared/Common.h"	//cl shared header
#include <OpenCL/cl.hpp>

namespace HydroGPU {
namespace Solver {
struct Solver;

struct SelfGravitation {
protected:
	Solver* solver;

public:
	cl::Buffer potentialBuffer;

protected:
	cl::Kernel gravityPotentialPoissonRelaxKernel;
	cl::Kernel calcGravityDerivKernel;

public:
	SelfGravitation(Solver* solver);
	virtual void initBuffers();
	virtual void initKernels();
	virtual std::vector<std::string> getProgramSources();
	virtual void resetState(std::vector<real>& potentialVec, std::vector<real>& stateVec);
	virtual void applyPotential();
	virtual void potentialBoundary();
};

}
}
