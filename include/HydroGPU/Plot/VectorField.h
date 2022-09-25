#pragma once

#include "CLCommon/cl.hpp"
#include "GLCxx/Buffer.h"
#include "GLCxx/gl.h"
#include <memory>

namespace HydroGPU {
namespace Solver {
struct Solver;
}
namespace Plot {

struct VectorField {
protected:
	std::shared_ptr<HydroGPU::Solver::Solver> solver;
	GLCxx::Buffer glBuffer;
	
	//with cl/gl sharing
	cl::BufferGL vertexBufferGL;
	//without cl/gl sharing	
	cl::Buffer vertexBufferCL;
	std::vector<float> vertexBufferCPU;

	cl::Kernel updateVectorFieldKernel;
	int resolution = {};
	
	int vertexCount = 0;
public:
	VectorField(std::shared_ptr<HydroGPU::Solver::Solver> solver_, int resolution_);
	
	void display();
	
	int variable = 0;
	float scale = .125f;
	
	int getResolution() const { return resolution; }
};

}
}

