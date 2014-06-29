#include "HydroGPU/EulerRoe.h"
#include "Common/File.h"

std::vector<std::string> EulerRoe::getProgramSources() {
	std::vector<std::string> sources = Super::getProgramSources();
	sources.push_back(Common::File::read("EulerRoe.cl"));
	return sources;
}

