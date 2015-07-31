#pragma once

#include <OpenGL/gl.h>
#include <OpenCL/cl.hpp>
#include <memory>
#include <string>

namespace HydroGPU {
struct HydroGPUApp;
namespace Plot {

struct Plot {
public:	//protected:
	HydroGPU::HydroGPUApp* app;
	
	GLuint tex;
	cl::ImageGL texCLMem;		//data is written to this buffer before rendering
	cl::Kernel convertToTexKernel;

public:
	Plot(HydroGPU::HydroGPUApp* app_);
	virtual ~Plot();

	virtual void init();
	virtual void display() = 0;
	virtual void screenshot(const std::string& filename) = 0;

public://protected:
	virtual void convertVariableToTex(int displayVariable);
};

}
}

