#pragma once

#include "HydroGPU/Equation/Euler.h"
#include "HydroGPU/Equation/Maxwell.h"

namespace HydroGPU {
struct HydroGPUApp;
namespace Equation {

struct EMHD : public Equation {
	using Super = Equation;

	Euler euler;
	Maxwell maxwell;

	EMHD(HydroGPUApp* app_);

	virtual void readStateCell(real* state, const real* source);
	virtual int stateGetBoundaryKernelForBoundaryMethod(int dim, int state, int minmax);
	virtual std::string name() const { return "EMHD"; }
};

}
}
