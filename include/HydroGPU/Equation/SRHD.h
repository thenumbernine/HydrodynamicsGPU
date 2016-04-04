#pragma once

#include "HydroGPU/Equation/Equation.h"
#include "HydroGPU/Equation/SelfGravitationBehavior.h"
#include <vector>
#include <string>

namespace HydroGPU {
struct HydroGPUApp;
namespace Equation {

struct SRHD : public Equation/*SelfGravitationBehavior<Equation>*/ {
	typedef Equation/*SelfGravitationBehavior<Equation>*/ Super;
	SRHD(HydroGPUApp* app_);
	virtual void getProgramSources(std::vector<std::string>& sources);
	virtual int stateGetBoundaryKernelForBoundaryMethod(int dim, int state, int minmax);
	virtual int numReadStateChannels();
	virtual std::string name() const { return "SRHD"; } 
};

}
}
