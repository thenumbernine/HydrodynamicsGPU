#pragma once

#include "HydroGPU/Solver/Solver.h"
#include <OpenCL/cl.hpp>

namespace HydroGPU {
namespace Solver {

struct MHDRemoveDivergence {
protected:
	Solver& solver;

public:	
	MHDRemoveDivergence(Solver& solver_);
	
	cl::Buffer magneticFieldDivergenceBuffer;
	cl::Buffer magneticFieldPotentialBuffer;
	
	cl::Kernel calcMagneticFieldDivergenceKernel;
	cl::Kernel magneticPotentialPoissonRelaxKernel;
	cl::Kernel magneticFieldRemoveDivergenceKernel;

	virtual void init();
	virtual void update();
	virtual void potentialBoundary();
	virtual void getProgramSources(std::vector<std::string>& sources);
};

}
}

