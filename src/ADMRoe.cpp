#include "HydroGPU/ADMRoe.h"
#include "HydroGPU/HydroGPUApp.h"
#include "Common/File.h"

std::vector<std::string> ADMRoe::getProgramSources() {
	std::vector<std::string> sources = Super::getProgramSources();
	real adm_BonaMasso_f = 1.f;
	app.lua.ref()["adm_BonaMasso_f"] >> adm_BonaMasso_f;
	sources[0] += 
	"#define ADM_BONA_MASSO_F " + toNumericString<real>(adm_BonaMasso_f) + "\n" +
	"enum {\n" +
	"\tSTATE_DX_LN_ALPHA,\n" +
	"\tSTATE_DX_LN_G,\n" +
	"};\n";
	sources.push_back(Common::File::read("ADMRoe.cl"));
	return sources;
}

