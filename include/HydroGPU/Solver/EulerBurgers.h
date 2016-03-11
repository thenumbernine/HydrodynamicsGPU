#pragma once

#include "HydroGPU/Solver/SelfGravitationBehavior.h"
#include "HydroGPU/Solver/FiniteVolumeSolver.h"
#include "Tensor/Vector.h"

namespace HydroGPU {
struct HydroGPUApp;
namespace Solver {

struct EulerBurgers : public SelfGravitationBehavior<FiniteVolumeSolver> {
	typedef SelfGravitationBehavior<FiniteVolumeSolver> Super;

protected:
	cl::Buffer interfaceVelocityBuffer;
	cl::Buffer pressureBuffer;

	cl::Kernel calcCellTimestepKernel;
	cl::Kernel calcInterfaceVelocityKernel;
	cl::Kernel computePressureKernel;
	cl::Kernel diffuseMomentumKernel;
	cl::Kernel diffuseWorkKernel;

public:
	EulerBurgers(HydroGPUApp* app);
protected:
	virtual void initKernels();
	virtual void initBuffers();
	virtual void createEquation();
	virtual std::vector<std::string> getProgramSources();
	virtual real calcTimestep();
	virtual void step(real dt);
};

}
}

