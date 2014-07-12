#include "HydroGPU/Roe.h"
#include "HydroGPU/HydroGPUApp.h"
#include "Common/File.h"

Roe::Roe(
	HydroGPUApp &app_)
: Super(app_)
, calcEigenBasisEvent("calcEigenBasis")
, calcCFLEvent("calcCFL")
, calcDeltaQTildeEvent("calcDeltaQTilde")
, calcFluxEvent("calcFlux")
, integrateFluxEvent("integrateFlux")
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
	entries.push_back(&integrateFluxEvent);

	//memory

	int volume = getVolume();

	eigenvaluesBuffer = clAlloc(sizeof(real) * numStates() * volume * app.dim);
	eigenvectorsBuffer = clAlloc(sizeof(real) * numStates() * numStates() * volume * app.dim);
	eigenvectorsInverseBuffer = clAlloc(sizeof(real) * numStates() * numStates() * volume * app.dim);
	deltaQTildeBuffer = clAlloc(sizeof(real) * numStates() * volume * app.dim);
	fluxBuffer = clAlloc(sizeof(real) * numStates() * volume * app.dim);

	{
		//zero interface and flux
		std::vector<real> zero(volume * app.dim * numStates());
		commands.enqueueWriteBuffer(fluxBuffer, CL_TRUE, 0, sizeof(real) * numStates() * volume * app.dim, &zero[0]);
	}

	calcEigenBasisKernel = cl::Kernel(program, "calcEigenBasis");
	app.setArgs(calcEigenBasisKernel, eigenvaluesBuffer, eigenvectorsBuffer, eigenvectorsInverseBuffer, stateBuffer, potentialBuffer);

	calcCFLKernel = cl::Kernel(program, "calcCFL");
	app.setArgs(calcCFLKernel, cflBuffer, eigenvaluesBuffer, app.cfl);
	
	calcDeltaQTildeKernel = cl::Kernel(program, "calcDeltaQTilde");
	app.setArgs(calcDeltaQTildeKernel, deltaQTildeBuffer, eigenvectorsInverseBuffer, stateBuffer);
	
	calcFluxKernel = cl::Kernel(program, "calcFlux");
	app.setArgs(calcFluxKernel, fluxBuffer, stateBuffer, eigenvaluesBuffer, eigenvectorsBuffer, eigenvectorsInverseBuffer, deltaQTildeBuffer, dtBuffer);
	
	integrateFluxKernel = cl::Kernel(program, "integrateFlux");
	app.setArgs(integrateFluxKernel, stateBuffer, fluxBuffer, dtBuffer);
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
	commands.enqueueNDRangeKernel(calcDeltaQTildeKernel, offsetNd, globalSize, localSize, nullptr, &calcDeltaQTildeEvent.clEvent);
	commands.enqueueNDRangeKernel(calcFluxKernel, offsetNd, globalSize, localSize, nullptr, &calcFluxEvent.clEvent);
	commands.enqueueNDRangeKernel(integrateFluxKernel, offsetNd, globalSize, localSize, nullptr, &integrateFluxEvent.clEvent);

	if (app.useGravity) {
		for (int i = 0; i < app.gaussSeidelMaxIter; ++i) {
			potentialBoundary();
			commands.enqueueNDRangeKernel(poissonRelaxKernel, offsetNd, globalSize, localSize);
		}
	
		commands.enqueueNDRangeKernel(addGravityKernel, offsetNd, globalSize, localSize);
		boundary();	
	}
}

