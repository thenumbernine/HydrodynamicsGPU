#pragma once

#include "HydroGPU/Solver3D.h"

struct HydroGPUApp;

struct EulerHLL : public Solver3D {
	typedef Solver3D Super;

	cl::Buffer eigenvaluesBuffer;
	cl::Buffer eigenvectorsBuffer;
	cl::Buffer eigenvectorsInverseBuffer;
	cl::Buffer deltaQTildeBuffer;
	cl::Buffer fluxBuffer;
	
	cl::Kernel calcFluxAndEigenvaluesKernel;
	cl::Kernel calcCFLKernel;
	cl::Kernel integrateFluxKernel;
	
	EventProfileEntry calcEigenBasisEvent;
	EventProfileEntry calcCFLEvent;
	EventProfileEntry integrateFluxEvent;
	
	EulerHLL(HydroGPUApp &app);
	virtual void init();

protected:
	virtual std::vector<std::string> getProgramSources();
	virtual void initStep();
	virtual void calcTimestep();
	virtual void step();
};

