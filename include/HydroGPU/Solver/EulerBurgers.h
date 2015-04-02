#pragma once

#include "HydroGPU/Solver/SelfGravitationBehavior.h"
#include "HydroGPU/Solver/Solver.h"
#include "Tensor/Vector.h"

namespace HydroGPU {
struct HydroGPUApp;
namespace Solver {

struct EulerBurgers : public SelfGravitationBehavior<Solver> {
	typedef SelfGravitationBehavior<Solver> Super;

protected:
	cl::Buffer interfaceVelocityBuffer;
	cl::Buffer fluxStateCoeffBuffer;
	cl::Buffer derivStateCoeffBuffer;
	cl::Buffer pressureBuffer;

	cl::Kernel calcCFLKernel;
	cl::Kernel calcInterfaceVelocityKernel;
	cl::Kernel calcFluxKernel;
	cl::Kernel calcDerivCoeffsFromFluxCoeffsKernel;
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

public:
	EulerBurgers(HydroGPUApp* app);
	virtual void init();
protected:
	virtual void initKernels();
	virtual void initBuffers();
	virtual void createEquation();
	virtual std::vector<std::string> getProgramSources();
	virtual void calcTimestep();
	virtual void step();
};

}
}

