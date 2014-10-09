#pragma once

#include "HydroGPU/Solver/Roe.h"

namespace HydroGPU {
struct HydroGPUApp;
namespace Solver {

/*
Roe solver for MHD equations
*/
struct MHDRoe : public Roe {
	typedef Roe Super;
	MHDRoe(HydroGPUApp& app);
protected:
	virtual std::vector<std::string> getProgramSources();
	virtual void init();
	virtual void initStep();
	virtual void calcFlux();

	//whether the flux has been written this frame or not
	cl::Buffer fluxFlagBuffer;

	cl::Kernel calcMHDFluxKernel;
};

}
}

