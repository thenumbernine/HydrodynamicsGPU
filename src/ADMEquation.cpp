#include "HydroGPU/ADMEquation.h"
#include "HydroGPU/Solver.h"
#include "HydroGPU/HydroGPUApp.h"
#include "Common/File.h"

ADMEquation::ADMEquation(Solver& solver) {
	numStates = 3;
}

void ADMEquation::getProgramSources(Solver& solver, std::vector<std::string>& sources) {
	real adm_BonaMasso_f = 1.f;
	solver.app.lua.ref()["adm_BonaMasso_f"] >> adm_BonaMasso_f;
	
	sources[0] += std::string() +
		"#define ADM_BONA_MASSO_F " + toNumericString<real>(adm_BonaMasso_f) + "\n" +
		"enum {\n" +
		"\tSTATE_DX_LN_ALPHA,\n" +
		"\tSTATE_DX_LN_G,\n" +
		"\tSTATE_K_TILDE,\n" +
		"};\n";
	
	sources.push_back(Common::File::read("ADMCommon.cl"));
}
