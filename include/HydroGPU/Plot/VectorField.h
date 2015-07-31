#pragma once

#include <OpenCL/cl.hpp>
#include <OpenGL/gl.h>
#include <memory>

namespace HydroGPU {
namespace Solver {
struct Solver;
}
namespace Plot {

struct VectorField {
protected:
	std::shared_ptr<HydroGPU::Solver::Solver> solver;
	GLuint vectorFieldGLBuffer;
	cl::BufferGL vectorFieldVertexBuffer;
	cl::Kernel updateVectorFieldKernel;
	int vectorFieldResolution;
	int vectorFieldVertexCount;
public:
	VectorField(std::shared_ptr<HydroGPU::Solver::Solver> solver_);
	virtual ~VectorField();
	virtual void display();
};

}
}

