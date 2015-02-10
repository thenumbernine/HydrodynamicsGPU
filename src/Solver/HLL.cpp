#include "HydroGPU/Solver/HLL.h"
#include "HydroGPU/HydroGPUApp.h"
#include "Common/File.h"

namespace HydroGPU {
namespace Solver {

void HLL::init() {
	Super::init();

	cl::Context context = app.context;

	//memory

	int volume = getVolume();

	eigenvaluesBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(real) * numStates() * volume * app.dim);
	fluxBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(real) * numStates() * volume * app.dim);
	
	calcEigenvaluesKernel = cl::Kernel(program, "calcEigenvalues");
	app.setArgs(calcEigenvaluesKernel, eigenvaluesBuffer, stateBuffer, potentialBuffer);
	
	calcFluxKernel = cl::Kernel(program, "calcFlux");
	app.setArgs(calcFluxKernel, fluxBuffer, stateBuffer, eigenvaluesBuffer, potentialBuffer, dtBuffer);

	calcCFLKernel = cl::Kernel(program, "calcCFL");
	app.setArgs(calcCFLKernel, cflBuffer, eigenvaluesBuffer, app.cfl);
	
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

void HLL::calcTimestep() {
	commands.enqueueNDRangeKernel(calcCFLKernel, offsetNd, globalSize, localSize);
	findMinTimestep();	
}

void HLL::step() {
	integrator->integrate([&](cl::Buffer derivBuffer) {
		commands.enqueueNDRangeKernel(calcFluxKernel, offsetNd, globalSize, localSize);
		calcFluxDerivKernel.setArg(0, derivBuffer);
		commands.enqueueNDRangeKernel(calcFluxDerivKernel, offsetNd, globalSize, localSize);
	});

	applyPotential();
}

}
}


