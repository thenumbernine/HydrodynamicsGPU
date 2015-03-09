#include "HydroGPU/Solver/MaxwellRoe.h"
#include "HydroGPU/HydroGPUApp.h"
#include "HydroGPU/Equation/Maxwell.h"

namespace HydroGPU {
namespace Solver {

void MaxwellRoe::init() {
	Super::init();
	addSourceKernel = cl::Kernel(program, "addSource");
	addSourceKernel.setArg(1, stateBuffer);
}
	
void MaxwellRoe::createEquation() {
	equation = std::make_shared<HydroGPU::Equation::Maxwell>(this);
}

std::vector<std::string> MaxwellRoe::getProgramSources() {
	std::vector<std::string> sources = Super::getProgramSources();
	sources.push_back("#include \"MaxwellRoe.cl\"\n");
	return sources;
}

//zero cell-based information required.
// unless you want to store permittivity and permeability.
// would a dynamic permittivity and permeability affect the flux equations?
int MaxwellRoe::getEigenTransformStructSize() {
	//how will OpenCL respond to an allocation of zero bytes?
	//not well...
	return 1;
}

std::vector<std::string> MaxwellRoe::getEigenProgramSources() {
	return {};
}

void MaxwellRoe::calcDeriv(cl::Buffer derivBuffer) {
	Super::calcDeriv(derivBuffer);
	addSourceKernel.setArg(0, derivBuffer);
	commands.enqueueNDRangeKernel(addSourceKernel, offsetNd, globalSize, localSize);
}

}
}

