#include "HydroGPU/MHDEquation.h"
#include "HydroGPU/HydroGPUApp.h"
#include "HydroGPU/Solver.h"
#include "Common/File.h"

MHDEquation::MHDEquation(Solver& solver) {
	numStates = 8;
}

std::string MHDEquation::getSource(Solver& solver) {
	std::string source = std::string() +
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
	
	source += Common::File::read("EulerMHDCommon.cl");

	return source;
}

