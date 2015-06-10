#pragma once

#include "HydroGPU/Solver/Roe.h"

namespace HydroGPU {
namespace Solver {

/*
Roe solver for ADM3D equations
*/
struct ADM3DRoe : public Roe {
	typedef Roe Super;
	using Super::Super;

protected:
	cl::Kernel addSourceKernel;

	virtual void initKernels();
	virtual void createEquation();
	virtual std::vector<std::string> getProgramSources();
	virtual std::vector<std::string> getEigenProgramSources();
	virtual int getEigenTransformStructSize();
	virtual void step(real dt);
};

}
}


