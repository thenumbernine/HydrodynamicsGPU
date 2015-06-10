#pragma once

#include "HydroGPU/Solver/Solver.h"

namespace HydroGPU {
struct HydroGPUApp;
namespace Solver {

/*
General Roe solver
subclasses need to implement calcEigenBasisSide
*/
struct Roe : public Solver {
	typedef Solver Super;

	cl::Buffer eigenvaluesBuffer;
	cl::Buffer eigenfieldsBuffer;	//contains forward and inverse transform information
	cl::Buffer deltaQTildeBuffer;
	cl::Buffer fluxBuffer;
	
	cl::Kernel calcEigenBasisSideKernel;
	cl::Kernel findMinTimestepKernel;
	cl::Kernel calcDeltaQTildeKernel;
	cl::Kernel calcFluxKernel;
	cl::Kernel calcFluxDerivKernel;
	
	EventProfileEntry calcEigenBasisSideEvent;
	EventProfileEntry findMinTimestepEvent;
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
	virtual void initFluxSide(int side);
	virtual real calcTimestep();
	virtual void step(real dt);
	virtual void calcDeriv(real dt, cl::Buffer derivBuffer, int side);
	virtual void calcFlux(real dt, int side);
};

}
}

