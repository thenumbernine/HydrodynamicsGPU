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
	cl::Kernel constrainKernel;

	virtual void initKernels();
	virtual void createEquation();
	virtual std::vector<std::string> getProgramSources();
	virtual std::vector<std::string> getCalcFluxDerivProgramSources();
	virtual std::vector<std::string> getEigenProgramSources();
	virtual int getEigenTransformStructSize();
	virtual int getEigenSpaceDim();
	virtual void step(real dt);
public:
	virtual std::string name() const { return "ADM3DRoe"; }
};

}
}
