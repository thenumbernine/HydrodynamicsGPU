#include "HydroGPU/ADMRoe.h"
#include "HydroGPU/HydroGPUApp.h"
#include "HydroGPU/Equation/ADM.h"
#include "Common/File.h"

ADMRoe::ADMRoe(HydroGPUApp& app_)
: Super(app_)
{
	equation = std::make_shared<HydroGPU::Equation::ADM>(*this);
}

std::vector<std::string> ADMRoe::getProgramSources() {
	std::vector<std::string> sources = Super::getProgramSources();
	sources.push_back(Common::File::read("ADMRoe.cl"));
	return sources;
}

