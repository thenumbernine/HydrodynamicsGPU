#include "HydroGPU/EulerEquation.h"
#include "HydroGPU/HydroGPUApp.h"
#include "HydroGPU/Solver.h"
#include "Common/File.h"

EulerEquation::EulerEquation(Solver& solver) {
	numStates = 2 + solver.app.dim;
}

std::string EulerEquation::getSource(Solver& solver) {
	real gamma = 1.4f;
	solver.app.lua.ref()["gamma"] >> gamma;
	
	std::string source = 
		"enum {\n"
		"\tSTATE_DENSITY,\n"
		"\tSTATE_VELOCITY_X,\n";
	if (solver.app.dim > 1) {
		source += "\tSTATE_VELOCITY_Y,\n";
	}
	if (solver.app.dim > 2) {
		source += "\tSTATE_VELOCITY_Z,\n";
	}
	source += 
		"\tSTATE_ENERGY_TOTAL\n"
		"};\n";
	
	source += "#define GAMMA " + toNumericString<real>(gamma) + "\n";

	source += Common::File::read("EulerMHDCommon.cl");

	return source;
}

