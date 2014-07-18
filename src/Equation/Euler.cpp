#include "HydroGPU/Equation/Euler.h"
#include "HydroGPU/HydroGPUApp.h"
#include "HydroGPU/Solver/Solver.h"
#include "Common/File.h"
#include "Common/Exception.h"

namespace HydroGPU {
namespace Equation {

enum {
	BOUNDARY_METHOD_PERIODIC,
	BOUNDARY_METHOD_MIRROR,
	BOUNDARY_METHOD_FREEFLOW,
	NUM_BOUNDARY_METHODS
};

Euler::Euler(HydroGPU::Solver::Solver& solver) 
: Super()
{
	displayMethods = std::vector<std::string>{
		"DENSITY",
		"VELOCITY",
		"PRESSURE",
		"POTENTIAL"
	};

	//matches above 
	boundaryMethods = std::vector<std::string>{
		"PERIODIC",
		"MIRROR",
		"FREEFLOW"
	};

	states.push_back("DENSITY");
	states.push_back("MOMENTUM_X");
	if (solver.app.dim > 1) states.push_back("MOMENTUM_Y");
	if (solver.app.dim > 2) states.push_back("MOMENTUM_Z");
	states.push_back("ENERGY_TOTAL");
}

void Euler::getProgramSources(HydroGPU::Solver::Solver& solver, std::vector<std::string>& sources) {
	Super::getProgramSources(solver, sources);
	
	sources[0] += "#include \"HydroGPU/Shared/Common.h\"\n";	//for real's definition
	
	real gamma = 1.4f;
	solver.app.lua.ref()["gamma"] >> gamma;
	sources[0] += "constant real gamma = " + toNumericString<real>(gamma) + ";\n";

	sources.push_back(Common::File::read("EulerMHDCommon.cl"));
}

int Euler::stateGetBoundaryKernelForBoundaryMethod(HydroGPU::Solver::Solver& solver, int dim, int state) {
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

int Euler::gravityGetBoundaryKernelForBoundaryMethod(HydroGPU::Solver::Solver& solver, int dim) {
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

}
}

