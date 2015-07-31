#include "HydroGPU/Plot/Plot1D2D.h"
#include "HydroGPU/Plot/CameraOrtho.h"
#include "HydroGPU/Solver/Solver.h"
#include "HydroGPU/HydroGPUApp.h"
#include "Image/System.h"
#include "Image/FITS_IO.h"
#include <OpenGL/gl.h>

namespace HydroGPU {
namespace Plot {

Plot1D2D::Plot1D2D(std::shared_ptr<HydroGPU::Solver::Solver> solver)
: Super(solver)
{
	int volume = solver->getVolume();
	
	//get a texture going for visualizing the output
	glGenTextures(1, &tex);
	glBindTexture(GL_TEXTURE_2D, tex);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	Tensor::Vector<int,3> glWraps(GL_TEXTURE_WRAP_S, GL_TEXTURE_WRAP_T, GL_TEXTURE_WRAP_R);
	//specific to Euler
	for (int i = 0; i < solver->app->dim; ++i) {
		switch (solver->app->boundaryMethods(i, 0)) {	//can't wrap one side and not the other, so just use the min 
		case 0://BOUNDARY_PERIODIC:
			glTexParameteri(GL_TEXTURE_2D, glWraps(i), GL_REPEAT);
			break;
		case 1://BOUNDARY_MIRROR:
		case 2://BOUNDARY_FREEFLOW:
			glTexParameteri(GL_TEXTURE_2D, glWraps(i), GL_CLAMP_TO_EDGE);
			break;
		}
	}
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F_ARB, solver->app->size.s[0], solver->app->size.s[1], 0, GL_RGBA, GL_FLOAT, nullptr);
	solver->totalAlloc += sizeof(float) * 4 * volume;
	std::cout << "allocating texture size " << (sizeof(float) * 4 * volume) << " running total " << solver->totalAlloc << std::endl;
	glBindTexture(GL_TEXTURE_2D, 0);
	int err = glGetError();
	if (err != 0) throw Common::Exception() << "failed to create GL texture.  got error " << err;
}

void Plot1D2D::screenshot(const std::string& filename) {
	std::shared_ptr<Image::Image> image = std::make_shared<Image::Image>(
		Tensor::Vector<int,2>(solver->app->size.s[0], solver->app->size.s[1]),
		nullptr, 3);
	
	glBindTexture(GL_TEXTURE_2D, tex);
	glGetTexImage(GL_TEXTURE_2D, 0, GL_RGB, GL_UNSIGNED_BYTE, image->getData());
	glBindTexture(GL_TEXTURE_2D, 0);
	Image::system->write(filename, image);
}

}
}
