#include "HydroGPU/Solver/Roe.h"
#include "HydroGPU/HydroGPUApp.h"
#include "Common/File.h"

namespace HydroGPU {
namespace Solver {

Roe::Roe(
	HydroGPUApp &app_)
: Super(app_)
, calcEigenBasisEvent("calcEigenBasis")
, calcCFLEvent("calcCFL")
, calcDeltaQTildeEvent("calcDeltaQTilde")
, calcFluxEvent("calcFlux")
{
}

void Roe::init() {
	Super::init();
	
	cl::Context context = app.context;

	entries.push_back(&calcEigenBasisEvent);
	if (!app.useFixedDT) {
		entries.push_back(&calcCFLEvent);
	}
	entries.push_back(&calcDeltaQTildeEvent);
	entries.push_back(&calcFluxEvent);

	//memory

	int volume = getVolume();

	eigenvaluesBuffer = clAlloc(sizeof(real) * numStates() * volume * app.dim);
	eigenvectorsBuffer = clAlloc(sizeof(real) * numStates() * numStates() * volume * app.dim);
	eigenvectorsInverseBuffer = clAlloc(sizeof(real) * numStates() * numStates() * volume * app.dim);
	deltaQTildeBuffer = clAlloc(sizeof(real) * numStates() * volume * app.dim);
	fluxBuffer = clAlloc(sizeof(real) * numStates() * volume * app.dim);

	//zero flux
	commands.enqueueFillBuffer(fluxBuffer, 0.f, 0, sizeof(real) * numStates() * volume * app.dim);

	calcEigenBasisKernel = cl::Kernel(program, "calcEigenBasis");
	app.setArgs(calcEigenBasisKernel, eigenvaluesBuffer, eigenvectorsBuffer, eigenvectorsInverseBuffer, stateBuffer, potentialBuffer);

	calcCFLKernel = cl::Kernel(program, "calcCFL");
	app.setArgs(calcCFLKernel, cflBuffer, eigenvaluesBuffer, app.cfl);
	
	calcDeltaQTildeKernel = cl::Kernel(program, "calcDeltaQTilde");
	app.setArgs(calcDeltaQTildeKernel, deltaQTildeBuffer, eigenvectorsInverseBuffer, stateBuffer);
	
	calcFluxKernel = cl::Kernel(program, "calcFlux");
	app.setArgs(calcFluxKernel, fluxBuffer, stateBuffer, eigenvaluesBuffer, eigenvectorsBuffer, eigenvectorsInverseBuffer, deltaQTildeBuffer, dtBuffer);
	
	calcFluxDerivKernel = cl::Kernel(program, "calcFluxDeriv");
	calcFluxDerivKernel.setArg(1, fluxBuffer);
}	

std::vector<std::string> Roe::getProgramSources() {
	std::vector<std::string> sources = Super::getProgramSources();
	sources.push_back(Common::File::read("Roe.cl"));
	return sources;
}

void Roe::initStep() {
	commands.enqueueNDRangeKernel(calcEigenBasisKernel, offsetNd, globalSize, localSize, nullptr, &calcEigenBasisEvent.clEvent);
}

void Roe::calcTimestep() {
	commands.enqueueNDRangeKernel(calcCFLKernel, offsetNd, globalSize, localSize, nullptr, &calcCFLEvent.clEvent);
	findMinTimestep();
}

void Roe::step() {
	integrator->integrate([&](cl::Buffer derivBuffer) {
		calcDeriv(derivBuffer);
	});

	applyPotential();
}

void Roe::calcFlux() {
	commands.enqueueNDRangeKernel(calcFluxKernel, offsetNd, globalSize, localSize, nullptr, &calcFluxEvent.clEvent);
}

void Roe::calcDeriv(cl::Buffer derivBuffer) {
	commands.enqueueNDRangeKernel(calcDeltaQTildeKernel, offsetNd, globalSize, localSize, nullptr, &calcDeltaQTildeEvent.clEvent);
	calcFlux();
	calcFluxDerivKernel.setArg(0, derivBuffer);
	commands.enqueueNDRangeKernel(calcFluxDerivKernel, offsetNd, globalSize, localSize);
}

}
}

