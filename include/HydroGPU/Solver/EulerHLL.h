#pragma once

#include "HydroGPU/Solver/Solver.h"

namespace HydroGPU {
struct HydroGPUApp;
namespace Solver {

struct EulerHLL : public Solver {
	typedef Solver Super;

	cl::Buffer eigenvaluesBuffer;
	cl::Buffer eigenvectorsBuffer;
	cl::Buffer eigenvectorsInverseBuffer;
	cl::Buffer deltaQTildeBuffer;
	cl::Buffer fluxBuffer;
	
	cl::Kernel calcEigenvaluesKernel;
	cl::Kernel calcFluxKernel;
	cl::Kernel calcCFLKernel;
	cl::Kernel calcFluxDerivKernel;
	
	EulerHLL(HydroGPUApp &app);
	virtual void init();

protected:
	virtual std::vector<std::string> getProgramSources();
	virtual std::string getFluxSource();
	virtual void initStep();
	virtual void calcTimestep();
	virtual void step();
};

}
}

