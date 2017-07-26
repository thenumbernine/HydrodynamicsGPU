#pragma once

#include "HydroGPU/Shared/Common.h"	//cl shared header
#include "CLCommon/cl.hpp"

namespace HydroGPU {
namespace Solver {
struct Solver;

struct SelfGravitation {
protected:
	Solver* solver;

public:
	cl::Buffer potentialBuffer;
	cl::Buffer solidBuffer;

protected:
	cl::Kernel gravityPotentialPoissonRelaxKernel;
	cl::Kernel calcGravityDerivKernel;

public:
	SelfGravitation(Solver* solver);
	virtual void initBuffers();
	virtual void initKernels();
	virtual std::vector<std::string> getProgramSources();
	virtual void resetState(std::vector<real>& stateVec, std::vector<real>& potentialVec, std::vector<char>& solidVec);
	virtual void applyPotential(real dt);
	virtual void potentialBoundary();

public:
	cl::Buffer getPotentialBuffer();
	cl::Buffer getSolidBuffer();
};

}
}
