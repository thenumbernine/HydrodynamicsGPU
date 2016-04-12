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
	GLuint glBuffer;
	cl::BufferGL vertexBuffer;
	cl::Kernel updateVectorFieldKernel;
	int resolution;
	int vertexCount;
public:
	VectorField(std::shared_ptr<HydroGPU::Solver::Solver> solver_, int resolution_);
	virtual ~VectorField();
	virtual void display();
	int variable;
	float scale;
	int getResolution() const { return resolution; }
};

}
}

