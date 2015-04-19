#pragma once

#include "HydroGPU/Integrator/Integrator.h"

namespace HydroGPU {
namespace Integrator {

struct BackwardEulerConjugateGradient : public Integrator {
	typedef Integrator Super;
	BackwardEulerConjugateGradient(HydroGPU::Solver::Solver* solver);
	virtual void integrate(std::function<void(cl::Buffer)> callback);
protected:
	cl::Buffer rBuffer, pBuffer, ApBuffer;
	cl::Kernel subtractKernel;
};

}
}

