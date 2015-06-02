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
	cl::Buffer eigenfieldsBuffer;	//contains forward and inverse transform information
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
	
	Roe(HydroGPUApp* app);
	virtual void init();
protected:
	virtual void initBuffers();
	virtual void initKernels();
	virtual std::vector<std::string> getProgramSources();
	virtual std::vector<std::string> getEigenProgramSources();
	virtual int getEigenSpaceDim();
	virtual int getEigenTransformStructSize();	//total size of forward and inverse
	virtual void initStep();
	virtual void calcTimestep();
	virtual void step();
	virtual void calcDeriv(cl::Buffer derivBuffer, int side);
	virtual void calcFlux(int side);
};

}
}

