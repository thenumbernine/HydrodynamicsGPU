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
	std::shared_ptr<Shader::Program> graphShader;

	Graph(HydroGPU::HydroGPUApp* app_);

	/*
	how to display variables ...
	*/
	struct Variable {
		bool enabled;
		float scale;
		int step;
		bool log;
		std::string name;
		Variable(const std::string& name_) : enabled(false), scale(1.f), step(1), log(false), name(name_) {}
	};
	std::vector<Variable> variables;

	virtual void display();
};

}
}
