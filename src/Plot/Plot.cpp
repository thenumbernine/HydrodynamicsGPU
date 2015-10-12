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

static int npo2(int x) {
	int y = 1;	
	for (--x; x>0; x>>=1, y<<=1);
	return y;
}

void Plot::convertVariableToTex(int displayVariable) {
	glFinish();
	cl::CommandQueue commands = app->solver->commands;

	std::vector<cl::Memory> acquireGLMems = {texCLMem};
	commands.enqueueAcquireGLObjects(&acquireGLMems);

	//if (app->clCommon->useGPU)
	{
		convertToTexKernel.setArg(1, displayVariable);
		
		//TODO round up next power of 2 of global size for texture ...
		cl::NDRange npo2size = 
			app->dim == 1
			? cl::NDRange(npo2(app->size.s[0]))
			: ( app->dim == 2
				? cl::NDRange(npo2(app->size.s[0]), npo2(app->size.s[1]))
				: ( app->dim == 3
					? cl::NDRange(npo2(app->size.s[0]), npo2(app->size.s[1]), npo2(app->size.s[2]))
					: throw Common::Exception() << "got an unknown dim " << app->dim
				)
			);
		//TODO is localSize compatible?  is it always 16x16 for 2D?
		commands.enqueueNDRangeKernel(convertToTexKernel, app->solver->offsetNd, npo2size /*app->solver->globalSize*/, app->solver->localSize);
	//} else {
		//TODO if we're not using GPU then we need to transfer the contents via a CPU buffer ... or not at all?
		//do the CL drivers correctly emulate the GL share writes when using CPU instead of GPU?
	//	throw Common::Exception() << "TODO";
	}

	commands.enqueueReleaseGLObjects(&acquireGLMems);
	commands.finish();
	
	int err = glGetError();
	if (err) std::cout << "GL error " << err << " at " << __FILE__ << ":" << __LINE__ << std::endl;
}

}
}
