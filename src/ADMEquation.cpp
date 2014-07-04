#include "HydroGPU/ADMEquation.h"
#include "HydroGPU/Solver.h"
#include "HydroGPU/HydroGPUApp.h"
#include "Common/File.h"

ADMEquation::ADMEquation(Solver& solver) 
: Super()
{
	numStates = 3;
	
	//TODO fixme
	displayMethods = std::vector<std::string>{
		"DENSITY",
		"VELOCITY",
		"PRESSURE",
		"MAGNETIC_FIELD",
		"GRAVITY_POTENTIAL"
	};
	
	boundaryMethods = std::vector<std::string>{
		"PERIODIC",
		"MIRROR",
		"FREEFLOW"
	};
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
	
	sources[0] += buildEnumCode(displayMethods);
	sources[0] += buildEnumCode(boundaryMethods);
	
	sources.push_back(Common::File::read("ADMCommon.cl"));
}
