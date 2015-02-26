#include "HydroGPU/Solver/BSSNOKRoe.h"
#include "HydroGPU/HydroGPUApp.h"
#include "HydroGPU/Equation/BSSNOK.h"

namespace HydroGPU {
namespace Solver {

void BSSNOKRoe::createEquation() {
	equation = std::make_shared<HydroGPU::Equation::BSSNOK>(this);
}

std::vector<std::string> BSSNOKRoe::getProgramSources() {
	std::vector<std::string> sources = Super::getProgramSources();
	sources.push_back("#include \"BSSNOKRoe.cl\"\n");
	return sources;
}

}
}

