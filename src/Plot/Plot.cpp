#include "HydroGPU/Plot/Plot.h"
#include "HydroGPU/Solver/Solver.h"
#include "HydroGPU/HydroGPUApp.h"

namespace HydroGPU {
namespace Plot {

Plot::Plot(HydroGPU::HydroGPUApp* app_)
: app(app_)
, tex(GLuint())
{
}

//call this after plot's subclass ctor, when the tex is allocated
void Plot::init() {
	//init buffers
	texCLMem = cl::ImageGL(app->clCommon->context, CL_MEM_WRITE_ONLY, GL_TEXTURE_3D, 0, tex);

	//init kernels
	convertToTexKernel = cl::Kernel(app->solver->program, "convertToTex");
	convertToTexKernel.setArg(0, texCLMem);
	convertToTexKernel.setArg(2, app->solver->stateBuffer);
	app->solver->setupConvertToTexKernelArgs();
}

Plot::~Plot() {
	glDeleteTextures(1, &tex);
}

void Plot::convertVariableToTex(int displayVariable) {
	glFinish();
	cl::CommandQueue commands = app->solver->commands;

	std::vector<cl::Memory> acquireGLMems = {texCLMem};
	commands.enqueueAcquireGLObjects(&acquireGLMems);

	if (app->clCommon->useGPU) {
		convertToTexKernel.setArg(1, displayVariable);
		commands.enqueueNDRangeKernel(convertToTexKernel, app->solver->offsetNd, app->solver->globalSize, app->solver->localSize);
	} else {
		//TODO if we're not using GPU then we need to transfer the contents via a CPU buffer ... or not at all?
		throw Common::Exception() << "TODO";
	}

	commands.enqueueReleaseGLObjects(&acquireGLMems);
	commands.finish();
	
	int err = glGetError();
	if (err) std::cout << "GL error " << err << " at " << __FILE__ << ":" << __LINE__ << std::endl;
}

}
}
