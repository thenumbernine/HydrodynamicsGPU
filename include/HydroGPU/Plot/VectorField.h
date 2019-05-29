#pragma once

#include "GLApp/gl.h"
#include "CLCommon/cl.hpp"
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
	
	//with cl/gl sharing
	cl::BufferGL vertexBufferGL;
	//without cl/gl sharing	
	cl::Buffer vertexBufferCL;
	std::vector<float> vertexBufferCPU;

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

