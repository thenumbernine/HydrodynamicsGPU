#pragma once

#include "HydroGPU/Solver/FiniteVolumeSolver.h"

namespace HydroGPU {
struct HydroGPUApp;
namespace Solver {

struct HLL : public FiniteVolumeSolver {
	using Super = FiniteVolumeSolver;

protected:
	cl::Buffer eigenvaluesBuffer;
	cl::Kernel calcEigenvaluesKernel;
	cl::Kernel calcCellTimestepKernel;

public:
	using Super::Super;

protected:
	virtual void initBuffers();
	virtual void initKernels();
	virtual std::vector<std::string> getProgramSources();
	virtual std::string getFluxSource() = 0;
	virtual void initStep();
	virtual real calcTimestep();
	virtual void step(real dt);
};

}
}

