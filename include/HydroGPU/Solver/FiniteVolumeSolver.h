#pragma once

#include "HydroGPU/Solver/Solver.h"

namespace HydroGPU {
struct HydroGPUApp;
namespace Solver {

struct FiniteVolumeSolver : public Solver {
protected:
	typedef Solver Super;
	
	cl::Buffer fluxBuffer;

	cl::Kernel calcFluxKernel;
	cl::Kernel calcFluxDerivKernel;

public:
	FiniteVolumeSolver(HydroGPUApp* app);

protected:
	virtual void initBuffers();
	virtual void initKernels();
	virtual std::vector<std::string> getProgramSources();
	virtual std::vector<std::string> getCalcFluxDerivProgramSources();
};

}
}
