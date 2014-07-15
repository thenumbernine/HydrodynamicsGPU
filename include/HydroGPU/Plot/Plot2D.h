#pragma once

#include "HydroGPU/Plot/Plot1D2D.h"

namespace HydroGPU {
namespace Solver {
struct Solver;
}
namespace Plot {

struct Plot2D : public Plot1D2D {
	typedef Plot1D2D Super;

	Plot2D(HydroGPU::Solver::Solver& solver);

	virtual void display();
};

}
}

