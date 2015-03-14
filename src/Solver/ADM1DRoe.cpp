#include "HydroGPU/Solver/ADM1DRoe.h"
#include "HydroGPU/HydroGPUApp.h"
#include "HydroGPU/Equation/ADM1D.h"

namespace HydroGPU {
namespace Solver {

void ADM1DRoe::createEquation() {
	equation = std::make_shared<HydroGPU::Equation::ADM1D>(this);
}

void ADM1DRoe::initKernels() {
	Super::initKernels();
	
	addSourceKernel = cl::Kernel(program, "addSource");
	addSourceKernel.setArg(1, stateBuffer);
}

std::vector<std::string> ADM1DRoe::getProgramSources() {
	std::vector<std::string> sources = Super::getProgramSources();
	sources.push_back("#include \"ADM1DRoe.cl\"\n");
	return sources;
}

int ADM1DRoe::getEigenTransformStructSize() {
	return 1;
}

std::vector<std::string> ADM1DRoe::getEigenProgramSources() {
	return {
		"enum {\n"
		"	EIGENFIELD_F\n"
		"};\n"
	};
}

void ADM1DRoe::calcDeriv(cl::Buffer derivBuffer) {
	Super::calcDeriv(derivBuffer);
	addSourceKernel.setArg(0, derivBuffer);
	commands.enqueueNDRangeKernel(addSourceKernel, offsetNd, globalSize, localSize);
}

}
}

