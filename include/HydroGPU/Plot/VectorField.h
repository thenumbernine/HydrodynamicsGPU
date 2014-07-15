#pragma once

#include <OpenCL/cl.hpp>
#include <OpenGL/gl.h>

namespace HydroGPU {
namespace Solver {
struct Solver;
}
namespace Plot {

struct VectorField {
	VectorField(HydroGPU::Solver::Solver& solver);
	virtual ~VectorField();
	virtual void display();

	HydroGPU::Solver::Solver& solver;
	GLuint vectorFieldGLBuffer;
	cl::BufferGL vectorFieldVertexBuffer;
	cl::Kernel updateVectorFieldKernel;
	int vectorFieldResolution;
	int vectorFieldVertexCount;
};

}
}

