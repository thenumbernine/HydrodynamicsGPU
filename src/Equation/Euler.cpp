#include "HydroGPU/Equation/Euler.h"
#include "HydroGPU/HydroGPUApp.h"
#include "HydroGPU/Boundary/Boundary.h"
#include "HydroGPU/Solver/Solver.h"
#include "Common/File.h"
#include "Common/Exception.h"

namespace HydroGPU {
namespace Equation {

Euler::Euler(HydroGPU::Solver::Solver* solver_) 
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

	states.push_back("DENSITY");
	states.push_back("MOMENTUM_X");
	if (solver->app->dim > 1) states.push_back("MOMENTUM_Y");
	if (solver->app->dim > 2) states.push_back("MOMENTUM_Z");
	states.push_back("ENERGY_TOTAL");
}

void Euler::getProgramSources(std::vector<std::string>& sources) {
	Super::getProgramSources(sources);
	
	sources[0] += "#include \"HydroGPU/Shared/Common.h\"\n";	//for real's definition
	
	real gamma = 1.4f;
	solver->app->lua.ref()["gamma"] >> gamma;
	sources[0] += "constant real gamma = " + toNumericString<real>(gamma) + ";\n";

	sources.push_back(Common::File::read("EulerMHDCommon.cl"));
}

int Euler::stateGetBoundaryKernelForBoundaryMethod(int dim, int state) {
	switch (solver->app->boundaryMethods(dim)) {
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
	throw Common::Exception() << "got an unknown boundary method " << solver->app->boundaryMethods(dim) << " for dim " << dim;
}

//Euler has special case to put MHD after velocity and put energy last
void Euler::readStateCell(real* state, const real* source) {
	state[0] = source[0];
	state[1] = source[1];
	if (solver->app->dim > 1) {
		state[2] = source[2];
	}
	if (solver->app->dim > 2) {
		state[3] = source[3];
	}
	if (states.size() == 8) {
		state[4] = source[4];
		state[5] = source[5];
		state[6] = source[6];
	}
	state[states.size()-1] = source[7];
}

int Euler::numReadStateChannels() {
	return 8;
}

}
}

