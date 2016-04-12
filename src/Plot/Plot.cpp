#include "HydroGPU/Plot/Plot.h"
#include "HydroGPU/Solver/Solver.h"
#include "HydroGPU/Equation/Equation.h"
#include "HydroGPU/HydroGPUApp.h"
#include "Image/System.h"
#include "Common/File.h"
#include <OpenGL/gl.h>
#include <iomanip>	//don't include iostream after this

namespace HydroGPU {
namespace Plot {

Plot::Plot(HydroGPU::HydroGPUApp* app_)
: app(app_)
, tex(GLuint())
{
	//get a texture going for visualizing the output
	glGenTextures(1, &tex);
	glBindTexture(GL_TEXTURE_2D, tex);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	Tensor::Vector<int,3> glWraps(GL_TEXTURE_WRAP_S, GL_TEXTURE_WRAP_T, GL_TEXTURE_WRAP_R);
	//specific to Euler
	for (int i = 0; i < app->dim; ++i) {
		switch (app->boundaryMethods(i, 0)) {	//can't wrap one side and not the other, so just use the min 
		case 0://BOUNDARY_PERIODIC:
			glTexParameteri(GL_TEXTURE_2D, glWraps(i), GL_REPEAT);
			break;
		case 1://BOUNDARY_MIRROR:
		case 2://BOUNDARY_FREEFLOW:
			glTexParameteri(GL_TEXTURE_2D, glWraps(i), GL_CLAMP_TO_EDGE);
			break;
		}
	}
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F_ARB, app->size.s[0], app->size.s[1], 0, GL_RGBA, GL_FLOAT, nullptr);
	app->solver->cl.totalAlloc += sizeof(float) * 4 * app->size.s[0] * app->size.s[1];
	std::cout << "allocating texture size " << (sizeof(float) * 4 * app->size.s[0] * app->size.s[1]) << " running total " << app->solver->cl.totalAlloc << std::endl;
	glBindTexture(GL_TEXTURE_2D, 0);
	int err = glGetError();
	if (err != 0) throw Common::Exception() << "failed to create GL texture.  got error " << err;
}

//call this after plot's subclass ctor, when the tex is allocated
void Plot::init() {
	//init buffers
	texCLMem = cl::ImageGL(app->clCommon->context, CL_MEM_WRITE_ONLY, GL_TEXTURE_3D, 0, tex);

	//init kernels
	convertToTexKernel = cl::Kernel(app->solver->program, "convertToTex");
	convertToTexKernel.setArg(0, texCLMem);
	app->solver->equation->setupConvertToTexKernelArgs(convertToTexKernel, app->solver.get());
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

void Plot::screenshot() {
	for (int i = 0; i < 10000; ++i) {
		std::stringstream ss;
		ss << "screenshot" << std::setw(5) << std::setfill('0') << i << ".png";
		std::string filename = ss.str();
		if (!Common::File::exists(filename)) {
			screenshotToFile(filename);
			return;
		}
	}
	throw Common::Exception() << "couldn't find an available filename";
}

void Plot::screenshotToFile(const std::string& filename) {
	::Tensor::Vector<int,2> screenSize = app->screenSize;

	std::shared_ptr<Image::Image> image = std::make_shared<Image::Image>(screenSize, nullptr, 3);
	glReadPixels(0, 0, screenSize(0), screenSize(1), GL_RGB, GL_UNSIGNED_BYTE, image->getData());
	
	//reverse rows
	std::shared_ptr<Image::Image> flipped = std::make_shared<Image::Image>(screenSize, nullptr, 3);
	for (int y = 0; y < screenSize(1); ++y) {
		memcpy(
			flipped->getData() + (screenSize(1)-y-1) * screenSize(0) * 3,
			image->getData() + y * screenSize(0) * 3,
			screenSize(0) * 3);
	}

	Image::system->write(filename, flipped);
}

}
}
