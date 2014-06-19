#pragma once

#include "HydroGPU/Solver2D.h"

struct HydroGPUApp;

struct HLLSolver2D : public Solver2D {
	typedef Solver2D Super;

	cl::Buffer eigenvaluesBuffer;
	cl::Buffer eigenvectorsBuffer;
	cl::Buffer eigenvectorsInverseBuffer;
	cl::Buffer deltaQTildeBuffer;
	cl::Buffer fluxBuffer;
	
	cl::Kernel calcEigenBasisKernel;
	cl::Kernel calcCFLKernel;
	cl::Kernel integrateFluxKernel;
	
	EventProfileEntry calcEigenBasisEvent;
	EventProfileEntry calcCFLEvent;
	EventProfileEntry integrateFluxEvent;
	
	HLLSolver2D(HydroGPUApp &app);

protected:
	virtual void initStep();
	virtual void calcTimestep();
	virtual void step();
};

