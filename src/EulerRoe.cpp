#include "HydroGPU/Equation/Euler.h"
#include "HydroGPU/EulerRoe.h"
#include "Common/File.h"

EulerRoe::EulerRoe(HydroGPUApp& app_)
: Super(app_)
{
	equation = std::make_shared<HydroGPU::Equation::Euler>(*this);
}

std::vector<std::string> EulerRoe::getProgramSources() {
	std::vector<std::string> sources = Super::getProgramSources();
	sources.push_back(Common::File::read("EulerRoe.cl"));
	return sources;
}

