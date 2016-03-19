#pragma once

#include "HydroGPU/Solver/Roe.h"

namespace HydroGPU {
namespace Solver {

/*
Roe solver for ADM 1D 5-variable equations
*/
struct ADM1DRoe : public Roe {
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
public:
	virtual std::string name() const { return "ADM1DRoe"; }
};

}
}
