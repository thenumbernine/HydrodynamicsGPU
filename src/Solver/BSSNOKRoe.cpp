#include "HydroGPU/Solver/BSSNOKRoe.h"
#include "HydroGPU/HydroGPUApp.h"
#include "HydroGPU/Equation/BSSNOK.h"
#include "Common/File.h"

namespace HydroGPU {
namespace Solver {

BSSNOKRoe::BSSNOKRoe(HydroGPUApp& app_)
: Super(app_)
{
	equation = std::make_shared<HydroGPU::Equation::BSSNOK>(*this);
}

std::vector<std::string> BSSNOKRoe::getProgramSources() {
	std::vector<std::string> sources = Super::getProgramSources();
	sources.push_back(Common::File::read("BSSNOKRoe.cl"));
	return sources;
}

}
}

