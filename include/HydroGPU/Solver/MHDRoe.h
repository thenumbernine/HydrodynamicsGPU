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
protected:
	typedef MHDRemoveDivergenceBehavior<SelfGravitationBehavior<Roe>> Super;

	//whether the flux has been written this frame or not
	cl::Buffer fluxFlagBuffer;

	cl::Kernel calcMHDFluxKernel;

public:
	using Super::Super;
	virtual void initBuffers();
	virtual void initKernels();
protected:
	virtual void createEquation();
	virtual std::vector<std::string> getProgramSources();
	virtual void initStep();
	virtual void calcFlux();
	virtual void step();
};

}
}

