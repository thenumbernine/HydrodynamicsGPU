#pragma once

#include "HydroGPU/Solver/SelfGravitationBehavior.h"
#include "HydroGPU/Solver/MHDRemoveDivergenceBehavior.h"
#include "HydroGPU/Solver/Roe.h"

namespace HydroGPU {
struct HydroGPUApp;
namespace Solver {

/*
Roe solver for MHD equations
*/
struct MHDRoe : public MHDRemoveDivergenceBehavior<SelfGravitationBehavior<Roe>> {
	typedef MHDRemoveDivergenceBehavior<SelfGravitationBehavior<Roe>> Super;
	using Super::Super;

protected:
	//whether the flux has been written this frame or not
	cl::Buffer fluxFlagBuffer;

	cl::Kernel calcMHDFluxKernel;

	virtual void initBuffers();
	virtual void initKernels();
	virtual void createEquation();
	virtual std::vector<std::string> getProgramSources();
	virtual void calcFlux(real dt);
	virtual void step(real dt);
	virtual void initFlux();
public:
	virtual std::string name() const { return "MHDRoe"; }
};

}
}
