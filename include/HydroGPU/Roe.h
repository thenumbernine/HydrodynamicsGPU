#pragma once

#include "HydroGPU/Solver3D.h"

struct HydroGPUApp;

/*
General Roe solver
Missing calcEigenBasis
*/
struct Roe : public Solver3D {
	typedef Solver3D Super;

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
	
	Roe(HydroGPUApp& app);
	virtual void init();
protected:
	virtual std::vector<std::string> getProgramSources();
	virtual void initStep();
	virtual void calcTimestep();
	virtual void step();
};

