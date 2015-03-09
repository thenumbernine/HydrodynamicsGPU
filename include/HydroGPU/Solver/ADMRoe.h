#pragma once

#include "HydroGPU/Solver/Roe.h"

namespace HydroGPU {
namespace Solver {

/*
Roe solver for ADM 1D 5-variable equations
*/
struct ADMRoe : public Roe {
	typedef Roe Super;

protected:
	cl::Kernel addSourceKernel;

public:
	using Super::Super;

protected:
	virtual void initKernels();
	virtual void createEquation();
	virtual std::vector<std::string> getProgramSources();
	virtual int getEigenTransformStructSize();
	virtual std::vector<std::string> getEigenProgramSources();
	virtual void calcDeriv(cl::Buffer derivBuffer);
};

}
}

