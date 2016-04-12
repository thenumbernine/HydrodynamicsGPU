#include "HydroGPU/Equation/MHD.h"
#include "HydroGPU/Solver/SelfGravitationBehavior.h"
#include "HydroGPU/Solver/MHDRemoveDivergenceBehavior.h"
#include "HydroGPU/toNumericString.h"
#include "HydroGPU/HydroGPUApp.h"
#include "Common/Exception.h"
#include <cassert>

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
		"MAGNETIC_DIVERGENCE_BUFFER",
		"MAGNETIC_DIVERGENCE_CALCULATED",
		"MAGNETIC_DIVERGENCE_ERROR",
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
	
	vectorFieldVars = {
		"VELOCITY",
		"MOMENTUM",
		"VORTICITY",
		"GRAVITY",
	};
}

void MHD::getProgramSources(std::vector<std::string>& sources) {
	Super::getProgramSources(sources);
	
	sources[0] += "#define MHD 1\n";	//for EulerMHDCommon.cl
	
	//precompute the sqrt
	sources[0] += std::string("#define mhd_sqrt_vacuumPermeability ") + toNumericString<real>(sqrt(app->lua["defs"]["mhd_vacuumPermeability"])) + std::string("\n");

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


void MHD::setupConvertToTexKernelArgs(cl::Kernel convertToTexKernel, Solver::Solver* solver) {
	Super::setupConvertToTexKernelArgs(convertToTexKernel, solver);

	Solver::SelfGravitationInterface* selfGravSolver = dynamic_cast<Solver::SelfGravitationInterface*>(solver);
	assert(selfGravSolver != nullptr);
	convertToTexKernel.setArg(3, selfGravSolver->getPotentialBuffer());
	convertToTexKernel.setArg(4, selfGravSolver->getSolidBuffer());
	
	Solver::MHDRemoveDivergenceInterface* mhdSolver = dynamic_cast<Solver::MHDRemoveDivergenceInterface*>(solver);
	assert(mhdSolver != nullptr);
	convertToTexKernel.setArg(5, mhdSolver->getMagneticFieldDivergenceBuffer());
}

void MHD::setupUpdateVectorFieldKernelArgs(cl::Kernel updateVectorFieldKernel, Solver::Solver* solver) {
	Super::setupUpdateVectorFieldKernelArgs(updateVectorFieldKernel, solver);

	Solver::SelfGravitationInterface* selfGravSolver = dynamic_cast<Solver::SelfGravitationInterface*>(solver);
	assert(selfGravSolver != nullptr);
	updateVectorFieldKernel.setArg(4, selfGravSolver->getPotentialBuffer());
}

}
}
