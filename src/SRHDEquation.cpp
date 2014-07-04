#include "HydroGPU/SRHDEquation.h"
#include "HydroGPU/HydroGPUApp.h"
#include "HydroGPU/Solver.h"
#include "Common/File.h"

SRHDEquation::SRHDEquation(Solver& solver) 
: Super()
{
	numStates = 2 + solver.app.dim;
	
	displayMethods = std::vector<std::string>{
		"DENSITY",
		"VELOCITY",
		"PRESSURE",
		"GRAVITY_POTENTIAL"
	};

	boundaryMethods = std::vector<std::string>{
		"PERIODIC",
		"MIRROR",
		"FREEFLOW"
	};
}

void SRHDEquation::getProgramSources(Solver& solver, std::vector<std::string>& sources) {
	real gamma = 1.4f;
	solver.app.lua.ref()["gamma"] >> gamma;
	
	sources[0] +=
		"enum {\n"
		"\tSTATE_REST_MASS_DENSITY,\n"
		"\tSTATE_MOMENTUM_DENSITY_X,\n";
	if (solver.app.dim > 1) {
		sources[0] += "\tSTATE_MOMENTUM_DENSITY_Y,\n";
	}
	if (solver.app.dim > 2) {
		sources[0] += "\tSTATE_MOMENTUM_DENSITY_Z,\n";
	}
	sources[0] += 
		"\tSTATE_ENERGY_DENSITY\n"
		"};\n";

	sources[0] += buildEnumCode(displayMethods);
	sources[0] += buildEnumCode(boundaryMethods);

	sources[0] += "#define GAMMA " + toNumericString<real>(gamma) + "\n";

	sources.push_back(Common::File::read("SRHDCommon.cl"));
}


