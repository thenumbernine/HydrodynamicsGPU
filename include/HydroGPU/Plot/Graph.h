#pragma once

#include <OpenGL/gl.h>

namespace HydroGPU {
namespace Solver {
struct Solver;
}
namespace Plot {

struct Graph {
	HydroGPU::Solver::Solver* solver;	
	GLuint tex;

	Graph(HydroGPU::Solver::Solver* solver_);
	virtual ~Graph();

	virtual void display();
};

}
}
