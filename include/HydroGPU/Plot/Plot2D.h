#pragma once

#include "HydroGPU/Plot/Plot1D2D.h"
#include "Shader/Program.h"
#include <memory>

namespace HydroGPU {
namespace Solver {
struct Solver;
}
namespace Plot {

struct Plot2D : public Plot1D2D {
	typedef Plot1D2D Super;
	
	std::shared_ptr<Shader::Program> heatShader;

	Plot2D(std::shared_ptr<HydroGPU::Solver::Solver> solver);

	virtual void display();
};

}
}

