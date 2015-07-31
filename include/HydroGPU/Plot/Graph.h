#pragma once

#include "Shader/Program.h"
#include <OpenGL/gl.h>
#include <memory>

namespace HydroGPU {
struct HydroGPUApp;
namespace Plot {

struct Graph {
	HydroGPU::HydroGPUApp* app;	
	float graphScale;
	std::shared_ptr<Shader::Program> graphShader;

	Graph(HydroGPU::HydroGPUApp* app_);

	virtual void display();
};

}
}
