#pragma once

#include "HydroGPU/Solver/Roe.h"

namespace HydroGPU {
namespace Solver {

/*
Roe solver for Euler equations
*/
struct EulerRoe : public Roe {
	typedef Roe Super;
	EulerRoe(HydroGPUApp& app);
protected:
	virtual std::vector<std::string> getProgramSources();
};

}
}

