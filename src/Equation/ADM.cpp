#include "HydroGPU/Equation/ADM.h"
#include "HydroGPU/Solver/Solver.h"
#include "HydroGPU/HydroGPUApp.h"
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

ADM::ADM(HydroGPU::Solver::Solver& solver) 
: Super()
{
	//TODO fixme
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

	states.push_back("DX_LN_ALPHA");
	states.push_back("DX_LN_G");
	states.push_back("K_TILDE");
}

void ADM::getProgramSources(HydroGPU::Solver::Solver& solver, std::vector<std::string>& sources) {
	Super::getProgramSources(solver, sources);
	
	real adm_BonaMasso_f = 1.f;
	solver.app.lua.ref()["adm_BonaMasso_f"] >> adm_BonaMasso_f;
	sources[0] += "#define ADM_BONA_MASSO_F " + toNumericString<real>(adm_BonaMasso_f) + "\n";
	
	sources.push_back(Common::File::read("ADMCommon.cl"));
}

int ADM::stateGetBoundaryKernelForBoundaryMethod(HydroGPU::Solver::Solver& solver, int dim, int state) {
	switch (solver.app.boundaryMethods(dim)) {
	case BOUNDARY_METHOD_PERIODIC:
		return BOUNDARY_KERNEL_PERIODIC;
		break;
	case BOUNDARY_METHOD_MIRROR:
		return BOUNDARY_KERNEL_MIRROR;	//which states should be negative'd and which shouldn't ...
		break;		
	case BOUNDARY_METHOD_FREEFLOW:
		return BOUNDARY_KERNEL_FREEFLOW;
		break;
	}
	throw Common::Exception() << "got an unknown boundary method " << solver.app.boundaryMethods(dim) << " for dim " << dim;
}

int ADM::gravityGetBoundaryKernelForBoundaryMethod(HydroGPU::Solver::Solver& solver, int dim) {
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

