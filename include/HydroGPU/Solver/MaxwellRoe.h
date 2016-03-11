#pragma once

#include "HydroGPU/Solver/Roe.h"

namespace HydroGPU {
namespace Solver {

/*
Roe solver for Maxwell equations
*/
struct MaxwellRoe : public Roe {
	typedef Roe Super;
	using Super::Super;

protected:	
	cl::Kernel addSourceKernel;

	virtual void initKernels();
	virtual void createEquation();
	virtual std::vector<std::string> getProgramSources();
	virtual int getEigenTransformStructSize();
	virtual std::vector<std::string> getEigenProgramSources();
	virtual void step(real dt);
};

}
}
