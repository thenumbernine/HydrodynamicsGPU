#include "HydroGPU/Solver/ADM3DRoe.h"
#include "HydroGPU/HydroGPUApp.h"
#include "HydroGPU/Equation/ADM3D.h"
#include "Common/File.h"

namespace HydroGPU {
namespace Solver {

void ADM3DRoe::createEquation() {
	equation = std::make_shared<HydroGPU::Equation::ADM3D>(this);
}

void ADM3DRoe::init() {
	Super::init();
	
	addSourceKernel = cl::Kernel(program, "addSource");
	addSourceKernel.setArg(1, stateBuffer);
}

std::vector<std::string> ADM3DRoe::getProgramSources() {
	std::vector<std::string> sources = Super::getProgramSources();
	sources.push_back(Common::File::read("ADM3DRoe.cl"));
	return sources;
}

void ADM3DRoe::calcDeriv(cl::Buffer derivBuffer) {
	Super::calcDeriv(derivBuffer);
	addSourceKernel.setArg(0, derivBuffer);
	commands.enqueueNDRangeKernel(addSourceKernel, offsetNd, globalSize, localSize);
}

}
}


