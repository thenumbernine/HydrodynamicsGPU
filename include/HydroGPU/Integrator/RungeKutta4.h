#pragma once

#include "HydroGPU/Integrator/Integrator.h"

namespace HydroGPU {
namespace Integrator {

struct RungeKutta4 : public Integrator {
	typedef Integrator Super;
	RungeKutta4(HydroGPU::Solver::Solver* solver);
	virtual void integrate(std::function<void(cl::Buffer)> callback);
protected:
	cl::Buffer restoreBuffer;
	cl::Buffer deriv1Buffer;
	cl::Buffer deriv2Buffer;
	cl::Buffer deriv3Buffer;
	cl::Buffer deriv4Buffer;
	cl::Kernel multAddKernel;
};

}
}

