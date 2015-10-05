#pragma once

#include "HydroGPU/Integrator/Integrator.h"
#include "HydroGPU/Shared/Common.h"	//cl shared header

namespace HydroGPU {
namespace Integrator {

struct BackwardEulerConjugateGradient : public Integrator {
	typedef Integrator Super;
	BackwardEulerConjugateGradient(HydroGPU::Solver::Solver* solver);
	virtual void integrate(real dt, std::function<void(cl::Buffer)> callback);
	
protected:
	cl::Buffer rBuffer;
	cl::Buffer pBuffer;
	cl::Buffer ApBuffer;
	cl::Buffer scratchScalarBuffer;

	cl::Kernel multAddKernel;
	cl::Kernel subtractKernel;
	cl::Kernel dotBufferKernel;

	void applyLinear(cl::Buffer result, cl::Buffer in, real dt);
	real dot(cl::Buffer a, cl::Buffer b, int length);
};

}
}

