#include "HydroGPU/Plot/Plot.h"
#include "HydroGPU/Solver/Solver.h"
#include "HydroGPU/Equation/Equation.h"
#include "HydroGPU/HydroGPUApp.h"
#include "Image/Image.h"
#include "GLCxx/gl.h"
#include "Common/File.h"
#include <iomanip>	//don't include iostream after this

namespace HydroGPU {
namespace Plot {

Plot::Plot(HydroGPU::HydroGPUApp* app_)
: app(app_)
{
	//get a texture going for visualizing the output
	if (app->dim == 1) {
		tex = GLCxx::Texture2D();	//1D?
	} else if (app->dim == 2) {
		tex = GLCxx::Texture2D();
	} else if (app->dim == 3) {
		tex = GLCxx::Texture3D();
	}

	tex
		.bind()
		.setParam<GL_TEXTURE_MIN_FILTER>(GL_NEAREST)
		.setParam<GL_TEXTURE_MAG_FILTER>(GL_LINEAR);
	
	Tensor::int3 glWraps(GL_TEXTURE_WRAP_S, GL_TEXTURE_WRAP_T, GL_TEXTURE_WRAP_R);
	//specific to Euler
	for (int i = 0; i < app->dim; ++i) {
		switch (app->boundaryMethods(i, 0)) {	//can't wrap one side and not the other, so just use the min 
		case 0://BOUNDARY_PERIODIC:
			tex.setParam(glWraps(i), GL_REPEAT);
			break;
		case 1://BOUNDARY_MIRROR:
		case 2://BOUNDARY_FREEFLOW:
			tex.setParam(glWraps(i), GL_CLAMP_TO_EDGE);
			break;
		}
	}
	
	if (app->dim == 1) {
		tex.create2D(app->size.s[0], app->size.s[1], GL_RGBA32F_ARB, GL_RGBA, GL_FLOAT);
	} else if (app->dim == 2) {
		tex.create2D(app->size.s[0], app->size.s[1], GL_RGBA32F_ARB, GL_RGBA, GL_FLOAT);
	} else if (app->dim == 3) {
		tex.create3D(app->size.s[0], app->size.s[1], app->size.s[2], GL_RGBA32F_ARB, GL_RGBA, GL_FLOAT);
	};
	int volume = app->solver->getVolume();
	app->solver->cl.totalAlloc += sizeof(float) * 4 * volume;
	std::cout << "allocating texture size " << (sizeof(float) * 4 * volume) << " running total " << app->solver->cl.totalAlloc << std::endl;
	tex.unbind();
	int err = glGetError();
	if (err != 0) throw Common::Exception() << "failed to create GL texture.  got error " << err;
}

//call this after plot's subclass ctor, when the tex is allocated
void Plot::init() {
	//init buffers
	//NOTICE tex is TEXTURE_2D
	//texCLMem is TEXTURE_3D
	//...and convertVariableToTex is image3d_t
	//...but it all seems to work.
	if (app->hasGLSharing) {
		texCLMem = cl::ImageGL(app->clCommon->context, CL_MEM_WRITE_ONLY, tex.target, 0, tex());
	} else {
		texBuffer = app->solver->cl.alloc(sizeof(float) * 4 * app->solver->getVolume());
	}

	//init kernels
	convertToTexKernel = cl::Kernel(app->solver->program, "convertToTex");
	app->solver->equation->setupConvertToTexKernelArgs(convertToTexKernel, app->solver.get());
}

Plot::~Plot() {}

static int npo2(int x) {
	int y = 1;	
	for (--x; x>0; x>>=1, y<<=1);
	return y;
}

void Plot::convertVariableToTex(int displayVariable) {
	glFinish();
	cl::CommandQueue commands = app->solver->commands;
	std::vector<cl::Memory> acquireGLMems = {texCLMem};

	if (app->hasGLSharing) {
		commands.enqueueAcquireGLObjects(&acquireGLMems);
	}

	if (app->hasGLSharing) {
		convertToTexKernel.setArg(0, texCLMem);
	} else {
		convertToTexKernel.setArg(0, texBuffer);
	}
		
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

	if (app->hasGLSharing) {
		commands.enqueueReleaseGLObjects(&acquireGLMems);
		commands.finish();
	} else {
		texVec.resize(4 * app->solver->getVolume());
		commands.enqueueReadBuffer(texBuffer, CL_TRUE, 0, sizeof(float) * 4 * app->solver->getVolume(), texVec.data());
		if (app->dim == 3) throw Common::Exception() << "still need to add 3D texture uploads with gl_sharing";
		tex
			.bind()
			.subImage2D(texVec.data())
			.unbind();
	}

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
	::Tensor::int2 screenSize = app->getScreenSize();

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
