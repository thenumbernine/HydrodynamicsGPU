#pragma once

#include <OpenCL/cl.hpp>
#include <functional>

struct Solver;

namespace HydroGPU {
namespace Integrator {

struct Integrator {
	Solver& solver;
	Integrator(Solver& solver);
	virtual void integrate(std::function<void(cl::Buffer)> callback) = 0;
};

}
}

