#include "HydroGPU/Equation/ADM1D.h"
#include "HydroGPU/Solver/Solver.h"
#include "HydroGPU/Boundary/Boundary.h"
#include "HydroGPU/HydroGPUApp.h"
#include "Common/File.h"
#include "Common/Exception.h"

namespace HydroGPU {
namespace Equation {

enum {
	BOUNDARY_METHOD_NONE = -1,
	BOUNDARY_METHOD_PERIODIC,
	BOUNDARY_METHOD_MIRROR,
	BOUNDARY_METHOD_FREEFLOW,
	NUM_BOUNDARY_METHODS
};

ADM1D::ADM1D(HydroGPU::Solver::Solver* solver_) 
: Super(solver_)
{
	displayVariables = std::vector<std::string>{
		"ALPHA",
		"G",
		"A",
		"D",
		"K_TILDE",
		"K"
	};

	//matches above
	boundaryMethods = std::vector<std::string>{
		"PERIODIC",
		"MIRROR",
		"FREEFLOW"
	};

	states.push_back("ALPHA");
	states.push_back("G");
	states.push_back("A");	// = dx ln alpha
	states.push_back("D");	// = dx ln g
	states.push_back("K_TILDE");	// = K sqrt(g)
}

void ADM1D::getProgramSources(std::vector<std::string>& sources) {
	Super::getProgramSources(sources);

	//TODO detect type, cast number to CL string or use literal string
	//if type is number ...
	//real adm_BonaMasso_f = 1.f;
	//solver->app->lua.ref()["adm_BonaMasso_f"] >> adm_BonaMasso_f;
	//sources[0] += "#define ADM_BONA_MASSO_F " + toNumericString<real>(adm_BonaMasso_f) + "\n";
	//else if type is string ...
	std::string adm_BonaMasso_f = "1.f";
	solver->app->lua.ref()["adm_BonaMasso_f"] >> adm_BonaMasso_f;
	sources[0] += "#define ADM_BONA_MASSO_F (" + adm_BonaMasso_f + ")\n";
	sources.push_back("#include \"ADM1DCommon.cl\"\n");
	
	//tell the Roe solver to calculate left & right separately
	// this is slower for dense small matrices (like the Euler equations)
	// but for the ADM, which hold no eigenfield struct data, and compute the eigentransform solely from state data
	// because they are sparse huge matrices, 
	//it saves both speed and memory.
	sources.push_back("#define ROE_EIGENFIELD_TRANSFORM_SEPARATE 1\n");
}

int ADM1D::stateGetBoundaryKernelForBoundaryMethod(int dim, int state, int minmax) {
	switch (solver->app->boundaryMethods(dim, minmax)) {
	case BOUNDARY_METHOD_NONE:
		return BOUNDARY_KERNEL_NONE;
	case BOUNDARY_METHOD_PERIODIC:
		return BOUNDARY_KERNEL_PERIODIC;
	case BOUNDARY_METHOD_MIRROR:
		return BOUNDARY_KERNEL_MIRROR;	//which states should be negative'd and which shouldn't ...
	case BOUNDARY_METHOD_FREEFLOW:
		return BOUNDARY_KERNEL_FREEFLOW;
	}
	throw Common::Exception() << "got an unknown boundary method " << solver->app->boundaryMethods(dim, minmax) << " for dim " << dim;
}

}
}

