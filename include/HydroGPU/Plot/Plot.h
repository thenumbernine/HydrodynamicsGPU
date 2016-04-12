#pragma once

#include <OpenGL/gl.h>
#include <OpenCL/cl.hpp>
#include <memory>
#include <string>

namespace HydroGPU {
struct HydroGPUApp;
namespace Plot {

struct Plot {
protected:
	HydroGPU::HydroGPUApp* app;
	cl::ImageGL texCLMem;		//data is written to this buffer before rendering
	
public:
	cl::Kernel convertToTexKernel;	//public so Solver can set the input args of this 
	GLuint tex;

public:
	Plot(HydroGPU::HydroGPUApp* app_);
	virtual ~Plot();
	virtual void init();

	virtual void convertVariableToTex(int displayVariable);
	
public:
	void screenshot();
protected:
	void screenshotToFile(const std::string& filename);
};

}
}

