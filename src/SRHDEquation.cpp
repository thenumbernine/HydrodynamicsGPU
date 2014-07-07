#include "HydroGPU/SRHDEquation.h"
#include "HydroGPU/HydroGPUApp.h"
#include "HydroGPU/Solver.h"
#include "Common/File.h"
#include "Common/Exception.h"

enum {
	BOUNDARY_METHOD_PERIODIC,
	BOUNDARY_METHOD_MIRROR,
	BOUNDARY_METHOD_FREEFLOW,
	NUM_BOUNDARY_METHODS
};

SRHDEquation::SRHDEquation(Solver& solver) 
: Super()
{
	numStates = 2 + solver.app.dim;
	
	displayMethods = std::vector<std::string>{
		"DENSITY",
		"VELOCITY",
		"PRESSURE",
		"GRAVITY_POTENTIAL"
	};

	//matches above
	boundaryMethods = std::vector<std::string>{
		"PERIODIC",
		"MIRROR",
		"FREEFLOW"
	};
}

void SRHDEquation::getProgramSources(Solver& solver, std::vector<std::string>& sources) {
	std::vector<std::string> states;
	states.push_back("REST_MASS_DENSITY");
	states.push_back("MOMENTUM_DENSITY_X");
	if (solver.app.dim > 1) states.push_back("MOMENTUM_DENSITY_Y");
	if (solver.app.dim > 2) states.push_back("MOMENTUM_DENSITY_Z");
	states.push_back("TOTAL_ENERGY_DENSITY");
	sources[0] += buildEnumCode("STATE", states);

	std::vector<std::string> primitives;
	primitives.push_back("DENSITY");
	primitives.push_back("VELOCITY_X");
	if (solver.app.dim > 1) primitives.push_back("VELOCITY_Y");
	if (solver.app.dim > 2) primitives.push_back("VELOCITY_Z");
	primitives.push_back("PRESSURE");
	sources[0] += buildEnumCode("PRIMITIVE", primitives);
	
	sources[0] += buildEnumCode("DISPLAY", displayMethods);
	sources[0] += buildEnumCode("BOUNDARY", boundaryMethods);
	
	real gamma = 1.4f;
	solver.app.lua.ref()["gamma"] >> gamma;
	sources[0] += "#define GAMMA " + toNumericString<real>(gamma) + "\n";
	
	sources.push_back(Common::File::read("SRHDCommon.cl"));
}

int SRHDEquation::stateGetBoundaryKernelForBoundaryMethod(Solver& solver, int dim, int state) {
	switch (solver.app.boundaryMethods(dim)) {
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
	throw Common::Exception() << "got an unknown boundary method " << solver.app.boundaryMethods(dim) << " for dim " << dim;
}

int SRHDEquation::gravityGetBoundaryKernelForBoundaryMethod(Solver& solver, int dim) {
	switch (solver.app.boundaryMethods(dim)) {
	case BOUNDARY_METHOD_PERIODIC:
		return BOUNDARY_KERNEL_PERIODIC;
		break;
	case BOUNDARY_METHOD_MIRROR:
		return BOUNDARY_KERNEL_FREEFLOW;
		break;		
	case BOUNDARY_METHOD_FREEFLOW:
		return BOUNDARY_KERNEL_FREEFLOW;
		break;
	}
	throw Common::Exception() << "got an unknown boundary method " << solver.app.boundaryMethods(dim) << " for dim " << dim;
}

