#pragma once

#include "GLApp/gl.h"
#include "CLCommon/cl.hpp"
#include <memory>
#include <string>

namespace HydroGPU {
struct HydroGPUApp;
namespace Plot {

struct Plot {
protected:
	HydroGPU::HydroGPUApp* app;
	
	cl::Kernel convertToTexKernel;

	GLuint target;
	GLuint tex;
	cl::ImageGL texCLMem;		//data is written to this buffer before rendering
	cl::Buffer texBuffer;		//if gl_sharing is missing then write it here before downloading and uploading it =p
	std::vector<float> texVec;	//so many copies ...

public:
	Plot(HydroGPU::HydroGPUApp* app_);
	virtual ~Plot();
	void init();

	//external call to convert the app->solver displayVariable (specified in solver->equation->displayVariables) to the tex 
	void convertVariableToTex(int displayVariable);
	
	//read-only so others can use the handle but not modify it
	//this is still not safe to others deleting/messing with its contents ...
	GLuint getTex() const { return tex; }
	GLuint getTarget() const { return target; }

public:
	void screenshot();
protected:
	void screenshotToFile(const std::string& filename);
};

}
}
