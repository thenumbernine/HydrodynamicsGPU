#include "HydroGPU/MHDEquation.h"
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

MHDEquation::MHDEquation(Solver& solver) 
: Super()
{
	displayMethods = std::vector<std::string>{
		"DENSITY",
		"VELOCITY",
		"PRESSURE",
		"MAGNETIC_FIELD",
		"GRAVITY_POTENTIAL"
	};

	//matches above
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

void MHDEquation::getProgramSources(Solver& solver, std::vector<std::string>& sources) {
	Super::getProgramSources(solver, sources);
	
	real gamma = 1.4f;
	solver.app.lua.ref()["gamma"] >> gamma;
	sources[0] += "#define GAMMA " + toNumericString<real>(gamma) + "\n";

	sources[0] += "#define MU0 1.f\n";	//TODO script me

	sources.push_back(Common::File::read("EulerMHDCommon.cl"));
	sources.push_back(Common::File::read("MHDCommon.cl"));
}

int MHDEquation::stateGetBoundaryKernelForBoundaryMethod(Solver& solver, int dim, int state) {
	switch (solver.app.boundaryMethods(dim)) {
	case BOUNDARY_METHOD_PERIODIC:
		return BOUNDARY_KERNEL_PERIODIC;
		break;
	case BOUNDARY_METHOD_MIRROR:
		return (dim + 1 == state || dim + 4 == state) ? BOUNDARY_KERNEL_REFLECT : BOUNDARY_KERNEL_MIRROR;
		break;		
	case BOUNDARY_METHOD_FREEFLOW:
		return BOUNDARY_KERNEL_FREEFLOW;
		break;
	}
	throw Common::Exception() << "got an unknown boundary method " << solver.app.boundaryMethods(dim) << " for dim " << dim;
}

int MHDEquation::gravityGetBoundaryKernelForBoundaryMethod(Solver& solver, int dim) {
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

