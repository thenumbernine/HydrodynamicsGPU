#include "HydroGPU/Solver/SRHDRoe.h"
#include "HydroGPU/Equation/SRHD.h"
#include "HydroGPU/HydroGPUApp.h"
#include "HydroGPU/Plot/Plot.h"

namespace HydroGPU {
namespace Solver {

void SRHDRoe::init() {
	Super::init();

	int volume = getVolume();
	primitiveBuffer = clAlloc(sizeof(real) * numStates() * volume);

	calcEigenBasisSideKernel.setArg(4, primitiveBuffer);
	//calcEigenBasisSideKernel.setArg(5, selfgrav->potentialBuffer);

	initVariablesKernel = cl::Kernel(program, "initVariables");
	CLCommon::setArgs(initVariablesKernel, stateBuffer, primitiveBuffer);
}

void SRHDRoe::setupConvertToTexKernelArgs() {
	app->plot->convertToTexKernel.setArg(2, primitiveBuffer);
}
	
void SRHDRoe::createEquation() {
	equation = std::make_shared<HydroGPU::Equation::SRHD>(this);
}

std::vector<std::string> SRHDRoe::getProgramSources() {
	std::vector<std::string> sources = Super::getProgramSources();
	sources.push_back("#include \"SRHDRoe.cl\"\n");
	return sources;
}

void SRHDRoe::resetState() {
	//store Newtonian Euler equation state variables in stateBuffer
	Super::resetState();
	commands.enqueueNDRangeKernel(initVariablesKernel, offsetNd, globalSize, localSize);
}

}
}

