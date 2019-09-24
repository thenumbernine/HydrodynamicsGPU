/*
runs an Euler sim and a Maxwell sim, and uses each's state variables as the other's source terms
*/
#pragma once

#include "HydroGPU/Solver/EulerRoe.h"
#include "HydroGPU/Solver/MaxwellRoe.h"

namespace HydroGPU {

struct HydroGPUApp;

namespace Equation {
struct Equation;
};

namespace Solver {

struct EMHDRoe : public ISolver {
protected:
	using Super = ISolver;

	EulerRoe euler;
	MaxwellRoe maxwell;
	std::shared_ptr<Equation::Equation> equation;

public:
	EMHDRoe(HydroGPUApp* app_);
	
	virtual std::string name() const { return "EMHDRoe"; }
	virtual std::shared_ptr<Equation::Equation> getEquation() const { return equation; }

	virtual void init();
	virtual void resetState();
	virtual void update();

protected:
	virtual void createEquation();
};

}
}
