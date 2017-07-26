#pragma once

#include "CLCommon/cl.hpp"

namespace HydroGPU {
namespace Solver {
struct Solver;

struct MHDRemoveDivergence {
protected:
	Solver* solver;
	
	cl::Buffer magneticFieldDivergenceBuffer;
	cl::Buffer magneticFieldPotentialBuffer;
	cl::Buffer magneticFieldPotential2Buffer;
	
	cl::Kernel calcMagneticFieldDivergenceKernel;
	cl::Kernel magneticPotentialPoissonRelaxKernel;
	cl::Kernel magneticFieldRemoveDivergenceKernel;

public:	
	MHDRemoveDivergence(Solver* solver_);

	virtual void init();
	virtual std::vector<std::string> getProgramSources();
	virtual void update();
	virtual void boundary(cl::Buffer buffer);

	cl::Buffer getMagneticFieldDivergenceBuffer();
};

}
}
