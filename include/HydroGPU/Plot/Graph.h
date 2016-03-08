#pragma once

#include "Shader/Program.h"
#include "Tensor/Tensor.h"
#include <OpenGL/gl.h>
#include <memory>

namespace HydroGPU {
struct HydroGPUApp;
namespace Plot {

struct Graph {
	HydroGPU::HydroGPUApp* app;	
	float scale;
	Tensor::Vector<int,3> step;
	std::vector<int> variables;
	std::shared_ptr<Shader::Program> graphShader;

	Graph(HydroGPU::HydroGPUApp* app_);

	virtual void display();
};

}
}
