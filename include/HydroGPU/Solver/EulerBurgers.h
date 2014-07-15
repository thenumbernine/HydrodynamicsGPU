#pragma once

#include "HydroGPU/Solver/Solver.h"
#include "Tensor/Vector.h"

namespace HydroGPU {
struct HydroGPUApp;
namespace Solver {

struct EulerBurgers : public Solver {
	typedef Solver Super;

	cl::Buffer interfaceVelocityBuffer;
	cl::Buffer fluxBuffer;
	cl::Buffer pressureBuffer;

	cl::Kernel calcCFLKernel;
	cl::Kernel calcInterfaceVelocityKernel;
	cl::Kernel calcFluxKernel;
	cl::Kernel calcFluxDerivKernel;
	cl::Kernel computePressureKernel;
	cl::Kernel diffuseMomentumKernel;
	cl::Kernel diffuseWorkKernel;
	
	EventProfileEntry calcCFLEvent;
	EventProfileEntry calcInterfaceVelocityEvent;
	EventProfileEntry calcFluxEvent;
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

}
}

