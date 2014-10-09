#pragma once

#include "HydroGPU/Solver/Solver.h"

namespace HydroGPU {
struct HydroGPUApp;
namespace Solver {

/*
General Roe solver
Missing calcEigenBasis
*/
struct Roe : public Solver {
	typedef Solver Super;

	cl::Buffer eigenvaluesBuffer;
	cl::Buffer eigenvectorsBuffer;
	cl::Buffer eigenvectorsInverseBuffer;
	cl::Buffer deltaQTildeBuffer;
	cl::Buffer fluxBuffer;
	
	cl::Kernel calcEigenBasisKernel;
	cl::Kernel calcCFLKernel;
	cl::Kernel calcDeltaQTildeKernel;
	cl::Kernel calcFluxKernel;
	cl::Kernel calcFluxDerivKernel;
	
	EventProfileEntry calcEigenBasisEvent;
	EventProfileEntry calcCFLEvent;
	EventProfileEntry calcDeltaQTildeEvent;
	EventProfileEntry calcFluxEvent;
	
	Roe(HydroGPUApp& app);
	virtual void init();
protected:
	virtual std::vector<std::string> getProgramSources();
	virtual void initStep();
	virtual void calcTimestep();
	virtual void step();
	virtual void calcDeriv(cl::Buffer derivBuffer);
	virtual void calcFlux();
};

}
}

