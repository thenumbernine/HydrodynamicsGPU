#include "HydroGPU/ADMRoe.h"
#include "Common/File.h"

std::vector<std::string> ADMRoe::getProgramSources() {
	std::vector<std::string> sources = Super::getProgramSources();
	sources.push_back(Common::File::read("ADMRoe.cl"));
	return sources;
}

