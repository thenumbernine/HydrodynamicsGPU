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
	
	cl::Kernel convertToTexKernel;
	GLuint tex;

public:
	Plot(HydroGPU::HydroGPUApp* app_);
	virtual ~Plot();
	virtual void init();

	//external call to convert the app->solver displayVariable (specified in solver->equation->displayVariables) to the tex 
	virtual void convertVariableToTex(int displayVariable);
	
	//read-only so others can use the handle but not modify it
	//this is still not safe to others deleting/messing with its contents ...
	GLuint getTex() { return tex; }

public:
	void screenshot();
protected:
	void screenshotToFile(const std::string& filename);
};

}
}
