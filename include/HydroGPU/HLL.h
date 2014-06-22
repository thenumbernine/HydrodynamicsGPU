#pragma once

#include "HydroGPU/Solver3D.h"

struct HydroGPUApp;

struct HLL : public Solver3D {
	typedef Solver3D Super;

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
	
	HLL(HydroGPUApp &app);

protected:
	virtual void initStep();
	virtual void calcTimestep();
	virtual void step();
};

