#pragma once

#include "HydroGPU/Plot/Plot.h"
#include "Shader/Program.h"
#include "Tensor/Quat.h"

namespace HydroGPU {
namespace Solver {
struct Solver;
}
namespace Plot {

struct Plot3D : public Plot {
	typedef Plot Super;
	
	Plot3D(HydroGPU::Solver::Solver& solver);
	
	virtual void display();
	virtual void resize();
	virtual void mousePan(int dx, int dy);
	virtual void mouseZoom(int dz);
	virtual void screenshot(const std::string& filename);
	
	Tensor::Quat<float> viewAngle;
	float viewDist;
	std::shared_ptr<Shader::Program> displayShader;
};

}
}

