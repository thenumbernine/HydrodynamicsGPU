#include "HydroGPU/Equation/MHD.h"
#include "HydroGPU/toNumericString.h"
#include "HydroGPU/HydroGPUApp.h"
#include "Common/Exception.h"

namespace HydroGPU {
namespace Equation {

MHD::MHD(HydroGPUApp* app_)
: Super(app_)
{
	displayVariables = std::vector<std::string>{
		"DENSITY",
		"VELOCITY",
		"PRESSURE",
		"MAGNETIC_FIELD",
		"MAGNETIC_DIVERGENCE",
		"POTENTIAL"
	};

	//matches Equations/SelfGravitationBehavior 
	boundaryMethods = std::vector<std::string>{
		"PERIODIC",
		"MIRROR",
		"FREEFLOW"
	};

	states = {
		"DENSITY",
		"MOMENTUM_X",
		"MOMENTUM_Y",
		"MOMENTUM_Z",
		"MAGNETIC_FIELD_X",
		"MAGNETIC_FIELD_Y",
		"MAGNETIC_FIELD_Z",
		"ENERGY_TOTAL"
	};
}

void MHD::getProgramSources(std::vector<std::string>& sources) {
	Super::getProgramSources(sources);
	
	sources[0] += "#include \"HydroGPU/Shared/Common.h\"\n";	//for real's definition
	
	real gamma = 1.4f;
	app->lua.ref()["gamma"] >> gamma;
	sources[0] += "constant real gamma = " + toNumericString<real>(gamma) + ";\n";

	real vaccuumPermeability = 1.f;
	app->lua.ref()["vaccuumPermeability"] >> vaccuumPermeability;
	sources[0] += "constant real vaccuumPermeability = " + toNumericString<real>(vaccuumPermeability) + ";\n";
	sources[0] += "constant real sqrtVaccuumPermeability = " + toNumericString<real>(sqrt(vaccuumPermeability)) + ";\n";

	//for EulerMHDCommon.cl
	sources[0] += "#define MHD 1\n";

	sources.push_back("#include \"MHDCommon.cl\"\n");
	sources.push_back("#include \"EulerMHDCommon.cl\"\n");
}

int MHD::stateGetBoundaryKernelForBoundaryMethod(int dim, int stateIndex, int minmax) {
	switch (app->boundaryMethods(dim, minmax)) {
	case BOUNDARY_METHOD_NONE:
		return BOUNDARY_KERNEL_NONE;
	case BOUNDARY_METHOD_PERIODIC:
		return BOUNDARY_KERNEL_PERIODIC;
	case BOUNDARY_METHOD_MIRROR:
		return (dim + 1 == stateIndex || dim + 4 == stateIndex) ? BOUNDARY_KERNEL_REFLECT : BOUNDARY_KERNEL_MIRROR;
	case BOUNDARY_METHOD_FREEFLOW:
		return BOUNDARY_KERNEL_FREEFLOW;
	}
	throw Common::Exception() << "got an unknown boundary method " << app->boundaryMethods(dim, minmax) << " for dim " << dim;
}

}
}
