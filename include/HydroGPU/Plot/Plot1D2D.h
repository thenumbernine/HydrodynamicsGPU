#pragma once

#include "HydroGPU/Plot/Plot.h"
#include <memory>

namespace HydroGPU {
namespace Solver {
struct Solver;
}
namespace Plot {

struct Plot1D2D : public Plot {
	typedef Plot Super;
	
	Plot1D2D(std::shared_ptr<HydroGPU::Solver::Solver> solver);
	
	virtual void screenshot(const std::string& filename);
};

}
}

