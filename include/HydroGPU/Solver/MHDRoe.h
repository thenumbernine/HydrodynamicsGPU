#pragma once

#include "HydroGPU/Solver/Roe.h"
#include "HydroGPU/Solver/MHDRemoveDivergence.h"

namespace HydroGPU {
struct HydroGPUApp;
namespace Solver {

/*
Roe solver for MHD equations
*/
struct MHDRoe : public Roe {
protected:
	typedef Roe Super;

	std::shared_ptr<MHDRemoveDivergence> divfree;

	//whether the flux has been written this frame or not
	cl::Buffer fluxFlagBuffer;

	cl::Kernel calcMHDFluxKernel;

public:
	using Super::Super;

protected:
	virtual void createEquation();
	virtual std::vector<std::string> getProgramSources();
	virtual void init();
	virtual void initStep();
	virtual void calcFlux();
	virtual void step();
};

}
}

