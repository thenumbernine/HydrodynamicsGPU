#include "HydroGPU/EulerEquation.h"
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

EulerEquation::EulerEquation(Solver& solver) 
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

void EulerEquation::getProgramSources(Solver& solver, std::vector<std::string>& sources) {
	std::vector<std::string> states;
	states.push_back("DENSITY");
	states.push_back("MOMENTUM_X");
	if (solver.app.dim > 1) states.push_back("MOMENTUM_Y");
	if (solver.app.dim > 2) states.push_back("MOMENTUM_Z");
	states.push_back("ENERGY_TOTAL");
	sources[0] += buildEnumCode("STATE", states);

	sources[0] += buildEnumCode("DISPLAY", displayMethods);
	sources[0] += buildEnumCode("BOUNDARY", boundaryMethods);
	
	real gamma = 1.4f;
	solver.app.lua.ref()["gamma"] >> gamma;
	sources[0] += "#define GAMMA " + toNumericString<real>(gamma) + "\n";

	sources.push_back(Common::File::read("EulerMHDCommon.cl"));
}

int EulerEquation::getBoundaryKernelForBoundaryMethod(Solver& solver, int dim, int state) {
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

