#pragma once

#include "HydroGPU/Shared/Common.h"	//cl shared header
#include "CLCommon/cl.hpp"
#include <functional>

namespace HydroGPU {
namespace Solver {
struct Solver;
}
namespace Integrator {

struct Integrator {
	HydroGPU::Solver::Solver* solver;
	Integrator(HydroGPU::Solver::Solver* solver);
	virtual void integrate(real dt, std::function<void(cl::Buffer)> callback) = 0;
};

}
}
