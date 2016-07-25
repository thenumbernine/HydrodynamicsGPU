#include "HydroGPU/Equation/Euler.h"
#include "HydroGPU/Solver/SelfGravitationBehavior.h"
#include "HydroGPU/Boundary/Boundary.h"
#include "HydroGPU/toNumericString.h"
#include "HydroGPU/HydroGPUApp.h"
#include "Common/Exception.h"
#include <cassert>

namespace HydroGPU {
namespace Equation {

Euler::Euler(HydroGPUApp* app_) 
: Super(app_)
{
	displayVariables = std::vector<std::string>{
		"DENSITY",
		"VELOCITY",
		"VELOCITY_X",
		"VELOCITY_Y",
		"VELOCITY_Z",
		"PRESSURE",
		"ENERGY_INTERNAL",
		"ENERGY_KINETIC",
		"ENERGY_TOTAL",
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
	if (app->dim > 1) states.push_back("MOMENTUM_Y");
	if (app->dim > 2) states.push_back("MOMENTUM_Z");
	states.push_back("ENERGY_TOTAL");

	vectorFieldVars = {
		"VELOCITY",
		"MOMENTUM",
		"GRAVITY",
		//"PRESSURE",
	};
	if (app->dim == 3) vectorFieldVars.push_back("VORTICITY");
}

void Euler::getProgramSources(std::vector<std::string>& sources) {
	Super::getProgramSources(sources);
	sources.push_back("#include \"EulerMHDCommon.cl\"\n");
}

int Euler::stateGetBoundaryKernelForBoundaryMethod(int dim, int state, int minmax) {
	switch (app->boundaryMethods(dim, minmax)) {
	case BOUNDARY_METHOD_NONE:
		return BOUNDARY_KERNEL_NONE;
	case BOUNDARY_METHOD_PERIODIC:
		return BOUNDARY_KERNEL_PERIODIC;
	case BOUNDARY_METHOD_MIRROR:
		return dim + 1 == state ? BOUNDARY_KERNEL_REFLECT : BOUNDARY_KERNEL_MIRROR;
	case BOUNDARY_METHOD_FREEFLOW:
		return BOUNDARY_KERNEL_FREEFLOW;
	}
	throw Common::Exception() << "got an unknown boundary method " << app->boundaryMethods(dim, minmax) << " for dim " << dim;
}

//Euler has special case to put MHD after velocity and put energy last
void Euler::readStateCell(real* state, const real* source) {
	state[0] = source[0];
	state[1] = source[1];
	if (app->dim > 1) {
		state[2] = source[2];
	}
	if (app->dim > 2) {
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

void Euler::setupConvertToTexKernelArgs(cl::Kernel convertToTexKernel, Solver::Solver* solver) {
	Super::setupConvertToTexKernelArgs(convertToTexKernel, solver);
	
	Solver::SelfGravitationInterface* selfGravSolver = dynamic_cast<Solver::SelfGravitationInterface*>(solver);
	assert(selfGravSolver != nullptr);
	convertToTexKernel.setArg(3, selfGravSolver->getPotentialBuffer());
	convertToTexKernel.setArg(4, selfGravSolver->getSolidBuffer());
}

void Euler::setupUpdateVectorFieldKernelArgs(cl::Kernel updateVectorFieldKernel, Solver::Solver* solver) {
	Super::setupUpdateVectorFieldKernelArgs(updateVectorFieldKernel, solver);

	Solver::SelfGravitationInterface* selfGravSolver = dynamic_cast<Solver::SelfGravitationInterface*>(solver);
	assert(selfGravSolver != nullptr);
	updateVectorFieldKernel.setArg(4, selfGravSolver->getPotentialBuffer());
}

}
}
