#pragma once

#include <OpenCL/cl.hpp>
#include <functional>

namespace HydroGPU {
namespace Solver {
struct Solver;
}
namespace Integrator {

struct Integrator {
	HydroGPU::Solver::Solver* solver;
	Integrator(HydroGPU::Solver::Solver* solver);
	virtual void integrate(std::function<void(cl::Buffer)> callback) = 0;

	//temporary while restructuring...
	virtual bool isImplicit() const { return false; }
};

}
}

