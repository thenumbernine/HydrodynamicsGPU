#include "HydroGPU/Plot/Plot.h"

namespace HydroGPU {
namespace Plot {

Plot::Plot(HydroGPU::Solver::Solver* solver_)
: solver(solver_)
, fluidTex(GLuint())
{
}

Plot::~Plot() {
	glDeleteTextures(1, &fluidTex);
}

}
}

