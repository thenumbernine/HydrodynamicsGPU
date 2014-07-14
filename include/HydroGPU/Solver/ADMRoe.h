#pragma once

#include "HydroGPU/Solver/Roe.h"

namespace HydroGPU {
namespace Solver {

/*
Roe solver for ADM equations
*/
struct ADMRoe : public Roe {
	typedef Roe Super;
	ADMRoe(HydroGPUApp& app);
protected:
	virtual std::vector<std::string> getProgramSources();
};

}
}

