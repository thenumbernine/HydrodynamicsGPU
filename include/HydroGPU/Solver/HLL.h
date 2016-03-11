#pragma once

#include "HydroGPU/Solver/FiniteVolumeSolver.h"

namespace HydroGPU {
struct HydroGPUApp;
namespace Solver {

struct HLL : public FiniteVolumeSolver {
	typedef FiniteVolumeSolver Super;

	cl::Buffer eigenvaluesBuffer;
	cl::Buffer eigenvectorsBuffer;
	cl::Buffer eigenvectorsInverseBuffer;
	cl::Buffer deltaQTildeBuffer;
	
	cl::Kernel calcEigenvaluesKernel;
	cl::Kernel calcCellTimestepKernel;
	
	using Super::Super;
	virtual void init();

protected:
	virtual std::vector<std::string> getProgramSources();
	virtual std::string getFluxSource() = 0;
	virtual void initStep();
	virtual real calcTimestep();
	virtual void step(real dt);
};

}
}

