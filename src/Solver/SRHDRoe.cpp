#include "HydroGPU/Solver/SRHDRoe.h"
#include "HydroGPU/Equation/SRHD.h"
#include "HydroGPU/HydroGPUApp.h"

namespace HydroGPU {
namespace Solver {

void SRHDRoe::init() {
	Super::init();

	int volume = getVolume();
	primitiveBuffer = clAlloc(sizeof(real) * numStates() * volume);

	initVariablesKernel = cl::Kernel(program, "initVariables");
	app->setArgs(initVariablesKernel, stateBuffer, primitiveBuffer);
	
	convertToTexKernel.setArg(0, primitiveBuffer);
	
	calcEigenBasisKernel.setArg(0, eigenvaluesBuffer);
	calcEigenBasisKernel.setArg(1, eigenvectorsBuffer);
	calcEigenBasisKernel.setArg(2, eigenvectorsInverseBuffer);
	calcEigenBasisKernel.setArg(3, primitiveBuffer);
	calcEigenBasisKernel.setArg(4, stateBuffer);
	//TODO get SRHD equation working with selfgrav by renaming STATE_REST_MASS_DENSITY to STATE_DENSITY
	//calcEigenBasisKernel.setArg(5, selfgrav->potentialBuffer);
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

