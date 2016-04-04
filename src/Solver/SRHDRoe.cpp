#include "HydroGPU/Solver/SRHDRoe.h"
#include "HydroGPU/Equation/SRHD.h"
#include "HydroGPU/HydroGPUApp.h"
#include "HydroGPU/Plot/Plot.h"

namespace HydroGPU {
namespace Solver {

void SRHDRoe::initBuffers() {
	Super::initBuffers();
	
	primitiveBuffer = cl.alloc(sizeof(real) * numStates() * getVolume());
}

void SRHDRoe::initKernels() {
	Super::initKernels();
	
	calcEigenBasisKernel.setArg(3, primitiveBuffer);
	
	initVariablesKernel = cl::Kernel(program, "initVariables");
	CLCommon::setArgs(initVariablesKernel, stateBuffer, primitiveBuffer);

	updatePrimitivesKernel = cl::Kernel(program, "updatePrimitives");
	CLCommon::setArgs(updatePrimitivesKernel, primitiveBuffer, stateBuffer);
}

//TODO make sure this runs when the plot or solver changes from the gui
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

/*
here's a dilemma ...
A single Forward Euler integration step takes place and the prims go out of sync.
Then you do root finding to re-update them.
What about multi-stage schemes?
The easy way is to just recompute prims before each flux integration.
But that is a lot of wasted updates.
It would be more efficient to save the prims along with the state vector.
That'd mean abstracting the push/pop functions of RK4...
*/
void SRHDRoe::calcDeriv(cl::Buffer derivBuffer, real dt) {
	commands.enqueueNDRangeKernel(updatePrimitivesKernel, offsetNd, globalSize, localSize);
	Super::calcDeriv(derivBuffer, dt);
}

}
}
