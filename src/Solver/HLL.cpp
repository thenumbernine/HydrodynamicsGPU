#include "HydroGPU/Solver/HLL.h"
#include "HydroGPU/HydroGPUApp.h"

namespace HydroGPU {
namespace Solver {

void HLL::init() {
	Super::init();

	cl::Context context = app->clCommon->context;

	//memory

	int volume = getVolume();

	eigenvaluesBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(real) * numStates() * volume * app->dim);
	fluxBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(real) * numStates() * volume * app->dim);
	
	calcEigenvaluesKernel = cl::Kernel(program, "calcEigenvalues");
	CLCommon::setArgs(calcEigenvaluesKernel, eigenvaluesBuffer, stateBuffer);
	
	calcFluxKernel = cl::Kernel(program, "calcFlux");
	CLCommon::setArgs(calcFluxKernel, fluxBuffer, stateBuffer, eigenvaluesBuffer);

	findMinTimestepKernel = cl::Kernel(program, "findMinTimestep");
	CLCommon::setArgs(findMinTimestepKernel, dtBuffer, eigenvaluesBuffer);
	
	calcFluxDerivKernel = cl::Kernel(program, "calcFluxDeriv");
	calcFluxDerivKernel.setArg(1, fluxBuffer);
}	

std::vector<std::string> HLL::getProgramSources() {
	std::vector<std::string> sources = Super::getProgramSources();
	sources.push_back(getFluxSource());
	return sources;
}

void HLL::initStep() {
	commands.enqueueNDRangeKernel(calcEigenvaluesKernel, offsetNd, globalSize, localSize);
}

real HLL::calcTimestep() {
	commands.enqueueNDRangeKernel(findMinTimestepKernel, offsetNd, globalSize, localSize);
	return findMinTimestep();	
}

void HLL::step(real dt) {
	calcFluxKernel.setArg(4, dt);
	integrator->integrate(dt, [&](cl::Buffer derivBuffer) {
		commands.enqueueNDRangeKernel(calcFluxKernel, offsetNd, globalSize, localSize);
		calcFluxDerivKernel.setArg(0, derivBuffer);
		commands.enqueueNDRangeKernel(calcFluxDerivKernel, offsetNd, globalSize, localSize);
	});
}

}
}


