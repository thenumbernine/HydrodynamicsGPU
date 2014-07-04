#include "HydroGPU/SRHDEquation.h"
#include "HydroGPU/SRHDRoe.h"
#include "Common/File.h"

SRHDRoe::SRHDRoe(HydroGPUApp& app_)
: Super(app_)
{
	equation = std::make_shared<SRHDEquation>(*this);
}

std::vector<std::string> SRHDRoe::getProgramSources() {
	std::vector<std::string> sources = Super::getProgramSources();
	sources.push_back(Common::File::read("SRHDRoe.cl"));
	return sources;
}

