#include "HydroGPU/Equation/Maxwell.h"
#include "HydroGPU/Boundary/Boundary.h"
#include "HydroGPU/toNumericString.h"
#include "HydroGPU/HydroGPUApp.h"
#include "Common/Exception.h"

namespace HydroGPU {
namespace Equation {

enum {
	BOUNDARY_METHOD_NONE = -1,
	BOUNDARY_METHOD_PERIODIC,
	BOUNDARY_METHOD_MIRROR,
	BOUNDARY_METHOD_FREEFLOW,
	NUM_BOUNDARY_METHODS
};

Maxwell::Maxwell(HydroGPUApp* app_) 
: Super(app_)
{
	displayVariables = std::vector<std::string>{
		"ELECTRIC",
		"ELECTRIC_X",
		"ELECTRIC_Y",
		"ELECTRIC_Z",
		"MAGNETIC",
		"MAGNETIC_X",
		"MAGNETIC_Y",
		"MAGNETIC_Z"
	};

	//matches above 
	boundaryMethods = std::vector<std::string>{
		"PERIODIC",
		"MIRROR",
		"FREEFLOW"
	};

	states = {
		"ELECTRIC_X",
		"ELECTRIC_Y",
		"ELECTRIC_Z",
		"MAGNETIC_X",
		"MAGNETIC_Y",
		"MAGNETIC_Z",
	};

	vectorFieldVars = {
		"ELECTRIC",
		"MAGNETIC",
		"POYNTING",
	};
}

void Maxwell::getProgramSources(std::vector<std::string>& sources) {
	Super::getProgramSources(sources);

	//precompute the sqrt
	for (std::string var : {"permittivity", "permeability", "conductivity"}) {
		sources[0] += std::string("#define maxwell_sqrt_") + var + std::string(" ") + toNumericString<real>(sqrt(app->lua["defs"]["maxwell_" + var])) + std::string("\n");
	}

	sources.push_back("#include \"MaxwellCommon.cl\"\n");
	
	//tell the Roe solver to calculate left & right separately
	// this is slower for dense small matrices (like the Euler equations)
	// but for the Maxwel, which hold no eigenvector struct data, and compute the eigentransform solely from state data
	// because they are sparse huge matrices, 
	//it saves both speed and memory.
	sources.push_back("#define ROE_EIGENFIELD_TRANSFORM_SEPARATE 1\n");
}

int Maxwell::stateGetBoundaryKernelForBoundaryMethod(int dim, int state, int minmax) {
	switch (app->boundaryMethods(dim, minmax)) {
	case BOUNDARY_METHOD_NONE:
		return BOUNDARY_KERNEL_NONE;
	case BOUNDARY_METHOD_PERIODIC:
		return BOUNDARY_KERNEL_PERIODIC;
	case BOUNDARY_METHOD_MIRROR:
		return (state == dim || state == 3+dim) ? BOUNDARY_KERNEL_REFLECT : BOUNDARY_KERNEL_MIRROR;
	case BOUNDARY_METHOD_FREEFLOW:
		return BOUNDARY_KERNEL_FREEFLOW;
	}
	throw Common::Exception() << "got an unknown boundary method " << app->boundaryMethods(dim, minmax) << " for dim " << dim;
}

}
}
