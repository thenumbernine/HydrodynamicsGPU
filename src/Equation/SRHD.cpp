#include "HydroGPU/Equation/SRHD.h"
#include "HydroGPU/Solver/Solver.h"
#include "HydroGPU/Boundary/Boundary.h"
#include "HydroGPU/HydroGPUApp.h"
#include "Common/File.h"
#include "Common/Exception.h"

namespace HydroGPU {
namespace Equation {

SRHD::SRHD(HydroGPU::Solver::Solver* solver_)
: Super(solver_)
{
	displayMethods = std::vector<std::string>{
		"DENSITY",
		"VELOCITY",
		"PRESSURE",
		"POTENTIAL"
	};

	//matches Equations/SelfGravitationBehavior 
	boundaryMethods = std::vector<std::string>{
		"PERIODIC",
		"MIRROR",
		"FREEFLOW"
	};

	states.push_back("REST_MASS_DENSITY");
	states.push_back("MOMENTUM_DENSITY_X");
	if (solver->app->dim > 1) states.push_back("MOMENTUM_DENSITY_Y");
	if (solver->app->dim > 2) states.push_back("MOMENTUM_DENSITY_Z");
	states.push_back("TOTAL_ENERGY_DENSITY");
}

void SRHD::getProgramSources(std::vector<std::string>& sources) {
	Super::getProgramSources(sources);

	std::vector<std::string> primitives;
	primitives.push_back("DENSITY");
	primitives.push_back("VELOCITY_X");
	if (solver->app->dim > 1) primitives.push_back("VELOCITY_Y");
	if (solver->app->dim > 2) primitives.push_back("VELOCITY_Z");
	primitives.push_back("PRESSURE");
	sources[0] += buildEnumCode("PRIMITIVE", primitives);
	
	sources[0] += "#include \"HydroGPU/Shared/Common.h\"\n";	//for real's definition
	
	real gamma = 1.4f;
	solver->app->lua.ref()["gamma"] >> gamma;
	sources[0] += "constant real gamma = " + toNumericString<real>(gamma) + ";\n";
	
	sources.push_back(Common::File::read("SRHDCommon.cl"));
}

int SRHD::stateGetBoundaryKernelForBoundaryMethod(int dim, int state, int minmax) {
	switch (solver->app->boundaryMethods(dim, minmax)) {
	case BOUNDARY_METHOD_PERIODIC:
		return BOUNDARY_KERNEL_PERIODIC;
		break;
	case BOUNDARY_METHOD_MIRROR:
		return dim + 1 == state ? BOUNDARY_KERNEL_REFLECT : BOUNDARY_KERNEL_MIRROR;
		break;		
	case BOUNDARY_METHOD_FREEFLOW:
		return BOUNDARY_KERNEL_FREEFLOW;
		break;
	}
	throw Common::Exception() << "got an unknown boundary method " << solver->app->boundaryMethods(dim, minmax) << " for dim " << dim;
}

int SRHD::numReadStateChannels() {
	return 8;
}

}
}

