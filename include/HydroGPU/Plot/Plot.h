#pragma once

#include <OpenGL/gl.h>
#include <string>

namespace HydroGPU {
namespace Solver {
struct Solver;
}
namespace Plot {

struct Plot {
	Plot(HydroGPU::Solver::Solver& solver);
	virtual ~Plot();

	virtual void display() = 0;
	virtual void resize() = 0;
	virtual void mousePan(int dx, int dy) = 0;
	virtual void mouseZoom(int dz) = 0;
	virtual void screenshot(const std::string& filename) = 0;
	
	HydroGPU::Solver::Solver& solver;
	GLuint fluidTex;
};

}
}

