#include "HydroGPU/Solver/ADMRoe.h"
#include "HydroGPU/HydroGPUApp.h"
#include "HydroGPU/Equation/ADM.h"

namespace HydroGPU {
namespace Solver {

void ADMRoe::createEquation() {
	equation = std::make_shared<HydroGPU::Equation::ADM>(this);
}

void ADMRoe::initKernels() {
	Super::initKernels();
	
	addSourceKernel = cl::Kernel(program, "addSource");
	addSourceKernel.setArg(1, stateBuffer);
}

std::vector<std::string> ADMRoe::getProgramSources() {
	std::vector<std::string> sources = Super::getProgramSources();
	sources.push_back("#include \"ADMRoe.cl\"\n");
	return sources;
}

int ADMRoe::getEigenfieldSize() {
	return 1;
}

std::vector<std::string> ADMRoe::getEigenfieldProgramSources() {
	return {
		"enum {\n"
		"	EIGENFIELD_F\n"
		"};\n"
	};
}

void ADMRoe::calcDeriv(cl::Buffer derivBuffer) {
	Super::calcDeriv(derivBuffer);
	addSourceKernel.setArg(0, derivBuffer);
	commands.enqueueNDRangeKernel(addSourceKernel, offsetNd, globalSize, localSize);
}

}
}

