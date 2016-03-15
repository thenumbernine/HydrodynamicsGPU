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
	
	/*
	Screenshot used to dump out the rgb values, back when I was converting them in the OpenCL kernel rather than in the shader.
	Now that that's done in the shader, the texture itself only holds the intensity of the channel.
	Therefore saving that texture is just saving a lower-resolution version of the FITS file.
	For this reason, I'm thinking the 'screenshot' function should be used for saving the literal screen shot.
	This way I can use this function for piecing together animations - especially of the heightmap graph view.
	*/
	virtual void screenshot(const std::string& filename);

public://protected:
	virtual void convertVariableToTex(int displayVariable);
};

}
}

