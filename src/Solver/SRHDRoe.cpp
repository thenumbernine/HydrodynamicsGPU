#include "HydroGPU/Solver/SRHDRoe.h"
#include "HydroGPU/Equation/SRHD.h"
#include "HydroGPU/HydroGPUApp.h"
#include "HydroGPU/Plot/Plot.h"

namespace HydroGPU {
namespace Solver {

void SRHDRoe::init() {
	Super::init();

	int volume = getVolume();
	primitiveBuffer = cl.alloc(sizeof(real) * numStates() * volume);

	calcEigenBasisKernel.setArg(3, primitiveBuffer);
	//calcEigenBasisKernel.setArg(4, selfgrav->potentialBuffer);

	initVariablesKernel = cl::Kernel(program, "initVariables");
	CLCommon::setArgs(initVariablesKernel, stateBuffer, primitiveBuffer);
}

void SRHDRoe::setupConvertToTexKernelArgs() {
	app->plot->convertToTexKernel.setArg(2, primitiveBuffer);
}
	
void SRHDRoe::createEquation() {
	equation = std::make_shared<HydroGPU::Equation::SRHD>(app);
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

