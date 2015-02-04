#include "HydroGPU/Equation/Maxwell.h"
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

Maxwell::Maxwell(HydroGPU::Solver::Solver& solver) 
: Super()
{
	displayMethods = std::vector<std::string>{
		"ELECTRIC",
		"MAGNETIC",
		"POTENTIAL"
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

void Maxwell::getProgramSources(HydroGPU::Solver::Solver& solver, std::vector<std::string>& sources) {
	Super::getProgramSources(solver, sources);
	
	sources[0] += "#include \"HydroGPU/Shared/Common.h\"\n";	//for real's definition

	real permeability = 1.f;
	solver.app.lua.ref()["permeability"] >> permeability;
	sources[0] += "constant real permeability = " + toNumericString<real>(permeability) + ";\n";
	sources[0] += "constant real sqrtPermeability = " + toNumericString<real>(sqrt(permeability)) + ";\n";
	
	real permittivity = 1.f;
	solver.app.lua.ref()["permittivity"] >> permittivity;
	sources[0] += "constant real permittivity = " + toNumericString<real>(permittivity) + ";\n";
	sources[0] += "constant real sqrtPermittivity = " + toNumericString<real>(sqrt(permittivity)) + ";\n";
	
	real conductivity = 1.f;
	solver.app.lua.ref()["conductivity"] >> conductivity;
	sources[0] += "constant real conductivity = " + toNumericString<real>(conductivity) + ";\n";
	sources[0] += "constant real sqrtConductivity = " + toNumericString<real>(sqrt(conductivity)) + ";\n";

	sources.push_back(Common::File::read("MaxwellCommon.cl"));
}

int Maxwell::stateGetBoundaryKernelForBoundaryMethod(HydroGPU::Solver::Solver& solver, int dim, int state) {
	switch (solver.app.boundaryMethods(dim)) {
	case BOUNDARY_METHOD_PERIODIC:
		return BOUNDARY_KERNEL_PERIODIC;
		break;
	case BOUNDARY_METHOD_MIRROR:
		return (state == dim || state == 3+dim) ? BOUNDARY_KERNEL_REFLECT : BOUNDARY_KERNEL_MIRROR;
		break;		
	case BOUNDARY_METHOD_FREEFLOW:
		return BOUNDARY_KERNEL_FREEFLOW;
		break;
	}
	throw Common::Exception() << "got an unknown boundary method " << solver.app.boundaryMethods(dim) << " for dim " << dim;
}

int Maxwell::gravityGetBoundaryKernelForBoundaryMethod(HydroGPU::Solver::Solver& solver, int dim) {
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

