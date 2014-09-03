#pragma once

#include "HydroGPU/Solver/Roe.h"

namespace HydroGPU {
struct HydroGPUApp;
namespace Solver {

/*
Roe solver for MHD equations
*/
struct MHDRoe : public Roe {
	typedef Roe Super;
	MHDRoe(HydroGPUApp& app);
protected:
	virtual std::vector<std::string> getProgramSources();
};

}
}

