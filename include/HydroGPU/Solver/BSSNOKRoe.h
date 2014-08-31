#pragma once

#include "HydroGPU/Solver/Roe.h"

namespace HydroGPU {
namespace Solver {

/*
Roe solver for BSSNOK equations
*/
struct BSSNOKRoe : public Roe {
	typedef Roe Super;
	BSSNOKRoe(HydroGPUApp& app);
protected:
	virtual std::vector<std::string> getProgramSources();
};

}
}


