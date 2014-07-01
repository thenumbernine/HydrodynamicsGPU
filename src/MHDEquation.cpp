#include "HydroGPU/MHDEquation.h"
#include "HydroGPU/HydroGPUApp.h"
#include "HydroGPU/Solver.h"
#include "Common/File.h"

MHDEquation::MHDEquation(Solver& solver) {
	numStates = 8;
}

void MHDEquation::getProgramSources(Solver& solver, std::vector<std::string>& sources) {
	real gamma = 1.4f;
	solver.app.lua.ref()["gamma"] >> gamma;
	
	sources[0] += 
	"enum {\n"
	"\tSTATE_DENSITY,\n"
	"\tSTATE_VELOCITY_X,\n"
	"\tSTATE_VELOCITY_Y,\n"
	"\tSTATE_VELOCITY_Z,\n"
	"\tSTATE_MAGNETIC_FIELD_X,\n"
	"\tSTATE_MAGNETIC_FIELD_Y,\n"
	"\tSTATE_MAGNETIC_FIELD_Z,\n"
	"\tSTATE_ENERGY_TOTAL,\n"
	"};\n";
	
	sources[0] += "#define GAMMA " + toNumericString<real>(gamma) + "\n";
	
	sources.push_back(Common::File::read("EulerMHDCommon.cl"));
}

