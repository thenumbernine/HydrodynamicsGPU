#include "HydroGPU/Equation/Maxwell.h"
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

Maxwell::Maxwell(HydroGPU::Solver::Solver* solver_) 
: Super(solver_)
{
	displayVariables = std::vector<std::string>{
		"ELECTRIC",
		"ELECTRIC_X",
		"ELECTRIC_Y",
		"ELECTRIC_Z",
		"MAGNETIC",
		"MAGNETIC_X",
		"MAGNETIC_Y",
		"MAGNETIC_Z"
	};

	//matches above 
	boundaryMethods = std::vector<std::string>{
		"PERIODIC",
		"MIRROR",
		"FREEFLOW"
	};

	states.push_back("ELECTRIC_X");
	states.push_back("ELECTRIC_Y");
	states.push_back("ELECTRIC_Z");
	states.push_back("MAGNETIC_X");
	states.push_back("MAGNETIC_Y");
	states.push_back("MAGNETIC_Z");
}

void Maxwell::getProgramSources(std::vector<std::string>& sources) {
	Super::getProgramSources(sources);
	
	sources[0] += "#include \"HydroGPU/Shared/Common.h\"\n";	//for real's definition

	real permeability = 1.f;
	solver->app->lua.ref()["permeability"] >> permeability;
	sources[0] += "constant real permeability = " + toNumericString<real>(permeability) + ";\n";
	sources[0] += "constant real sqrtPermeability = " + toNumericString<real>(sqrt(permeability)) + ";\n";
	
	real permittivity = 1.f;
	solver->app->lua.ref()["permittivity"] >> permittivity;
	sources[0] += "constant real permittivity = " + toNumericString<real>(permittivity) + ";\n";
	sources[0] += "constant real sqrtPermittivity = " + toNumericString<real>(sqrt(permittivity)) + ";\n";
	
	real conductivity = 1.f;
	solver->app->lua.ref()["conductivity"] >> conductivity;
	sources[0] += "constant real conductivity = " + toNumericString<real>(conductivity) + ";\n";
	sources[0] += "constant real sqrtConductivity = " + toNumericString<real>(sqrt(conductivity)) + ";\n";

	sources.push_back(Common::File::read("MaxwellCommon.cl"));
	
	//tell the Roe solver to calculate left & right separately
	// this is slower for dense small matrices (like the Euler equations)
	// but for the Maxwel, which hold no eigenvector struct data, and compute the eigentransform solely from state data
	// because they are sparse huge matrices, 
	//it saves both speed and memory.
	sources.push_back("#define ROE_EIGENFIELD_TRANSFORM_SEPARATE 1\n");
}

int Maxwell::stateGetBoundaryKernelForBoundaryMethod(int dim, int state, int minmax) {
	switch (solver->app->boundaryMethods(dim, minmax)) {
	case BOUNDARY_METHOD_NONE:
		return BOUNDARY_KERNEL_NONE;
	case BOUNDARY_METHOD_PERIODIC:
		return BOUNDARY_KERNEL_PERIODIC;
	case BOUNDARY_METHOD_MIRROR:
		return (state == dim || state == 3+dim) ? BOUNDARY_KERNEL_REFLECT : BOUNDARY_KERNEL_MIRROR;
	case BOUNDARY_METHOD_FREEFLOW:
		return BOUNDARY_KERNEL_FREEFLOW;
	}
	throw Common::Exception() << "got an unknown boundary method " << solver->app->boundaryMethods(dim, minmax) << " for dim " << dim;
}

}
}

