#include "HydroGPU/Solver/ADM3DRoe.h"
#include "HydroGPU/HydroGPUApp.h"
#include "HydroGPU/Equation/ADM3D.h"

namespace HydroGPU {
namespace Solver {

void ADM3DRoe::createEquation() {
	equation = std::make_shared<HydroGPU::Equation::ADM3D>(this);
}

void ADM3DRoe::initKernels() {
	Super::initKernels();
	
	addSourceKernel = cl::Kernel(program, "addSource");
	addSourceKernel.setArg(1, stateBuffer);
}

std::vector<std::string> ADM3DRoe::getProgramSources() {
	std::vector<std::string> sources = Super::getProgramSources();
	sources.push_back("#include \"ADM3DRoe.cl\"\n");
	return sources;
}

std::vector<std::string> ADM3DRoe::getEigenProgramSources() {
	return {};
}

int ADM3DRoe::getEigenTransformStructSize() {
	return numStates() + 6 + 1 + 1;	//states, gInv, g, f
}

void ADM3DRoe::step(real dt) {
	Super::step(dt);

	//see ADM1DRoe::step() for my thoughts on source and separabe integration
	integrator->integrate(dt, [&](cl::Buffer derivBuffer) {
		addSourceKernel.setArg(0, derivBuffer);
		commands.enqueueNDRangeKernel(addSourceKernel, offsetNd, globalSize, localSize);
	});
}

}
}


