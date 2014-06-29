#pragma once

#include "HydroGPU/Solver3D.h"
#include "Tensor/Vector.h"

struct HydroGPUApp;

struct EulerBurgers : public Solver3D {
	typedef Solver3D Super;

	cl::Buffer interfaceVelocityBuffer;
	cl::Buffer fluxBuffer;
	cl::Buffer pressureBuffer;

	cl::Kernel calcCFLKernel;
	cl::Kernel calcInterfaceVelocityKernel;
	cl::Kernel calcFluxKernel;
	cl::Kernel integrateFluxKernel;
	cl::Kernel computePressureKernel;
	cl::Kernel diffuseMomentumKernel;
	cl::Kernel diffuseWorkKernel;
	
	EventProfileEntry calcCFLEvent;
	EventProfileEntry calcInterfaceVelocityEvent;
	EventProfileEntry calcFluxEvent;
	EventProfileEntry integrateFluxEvent;
	EventProfileEntry computePressureEvent;
	EventProfileEntry diffuseMomentumEvent;
	EventProfileEntry diffuseWorkEvent;

	EulerBurgers(HydroGPUApp &app);
	virtual void init();	
protected:
	virtual std::vector<std::string> getProgramSources();
	virtual void calcTimestep();
	virtual void step();
};


