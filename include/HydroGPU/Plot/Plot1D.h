#pragma once

#include "HydroGPU/Plot/Plot1D2D.h"
#include "Shader/Program.h"

namespace HydroGPU {
namespace Solver {
struct Solver;
}
namespace Plot {

struct Plot1D : public Plot1D2D {
	typedef Plot1D2D Super;

	Plot1D(std::shared_ptr<HydroGPU::Solver::Solver> solver);

	virtual void display();

	std::shared_ptr<Shader::Program> displayShader;
};

}
}

