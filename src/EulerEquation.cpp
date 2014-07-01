#include "HydroGPU/EulerEquation.h"
#include "HydroGPU/HydroGPUApp.h"
#include "HydroGPU/Solver.h"
#include "Common/File.h"

EulerEquation::EulerEquation(Solver& solver) {
	numStates = 2 + solver.app.dim;
}

void EulerEquation::getProgramSources(Solver& solver, std::vector<std::string>& sources) {
	real gamma = 1.4f;
	solver.app.lua.ref()["gamma"] >> gamma;
	
	sources[0] +=
		"enum {\n"
		"\tSTATE_DENSITY,\n"
		"\tSTATE_VELOCITY_X,\n";
	if (solver.app.dim > 1) {
		sources[0] += "\tSTATE_VELOCITY_Y,\n";
	}
	if (solver.app.dim > 2) {
		sources[0] += "\tSTATE_VELOCITY_Z,\n";
	}
	sources[0] += 
		"\tSTATE_ENERGY_TOTAL\n"
		"};\n";
	
	sources[0] += "#define GAMMA " + toNumericString<real>(gamma) + "\n";

	sources.push_back(Common::File::read("EulerMHDCommon.cl"));
}

