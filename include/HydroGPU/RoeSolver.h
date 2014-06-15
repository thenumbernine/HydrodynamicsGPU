#pragma once

#include "HydroGPU/Solver2D.h"

struct HydroGPUApp;

struct RoeSolver : public Solver2D {
	typedef Solver2D Super;

	cl::Buffer eigenvaluesBuffer;
	cl::Buffer eigenvectorsBuffer;
	cl::Buffer eigenvectorsInverseBuffer;
	cl::Buffer deltaQTildeBuffer;
	cl::Buffer fluxBuffer;
	
	cl::Kernel calcEigenBasisKernel;
	cl::Kernel calcCFLKernel;
	cl::Kernel calcDeltaQTildeKernel;
	cl::Kernel calcFluxKernel;
	cl::Kernel integrateFluxKernel;
	
	EventProfileEntry calcEigenBasisEvent;
	EventProfileEntry calcCFLEvent;
	EventProfileEntry calcDeltaQTildeEvent;
	EventProfileEntry calcFluxEvent;
	EventProfileEntry integrateFluxEvent;
	
	RoeSolver(HydroGPUApp &app, std::vector<real4> stateVec);

protected:
	virtual void initStep();
	virtual void calcTimestep();
	virtual void step();
};

