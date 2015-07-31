#pragma once

#include "HydroGPU/Plot/Plot.h"
#include "Shader/Program.h"
#include <memory>

namespace HydroGPU {
namespace Solver {
struct Solver;
}
namespace Plot {

struct Plot3D : public Plot {
	typedef Plot Super;
	
	std::shared_ptr<Shader::Program> displayShader;

	Plot3D(std::shared_ptr<HydroGPU::Solver::Solver> solver);
	
	virtual void display();
	virtual void screenshot(const std::string& filename);	
};

}
}

