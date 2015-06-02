#pragma once

#include "HydroGPU/Solver/SelfGravitationBehavior.h"
#include "HydroGPU/Solver/Roe.h"

namespace HydroGPU {
namespace Solver {

/*
Roe solver for Euler equations
*/
struct EulerRoe : public SelfGravitationBehavior<Roe> {
	typedef SelfGravitationBehavior<Roe> Super;
	using Super::Super;
protected:
	virtual void initKernels();
	virtual void createEquation();
	virtual std::vector<std::string> getProgramSources();
	virtual void step();
};

}
}

