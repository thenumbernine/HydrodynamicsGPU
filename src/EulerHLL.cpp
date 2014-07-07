#include "HydroGPU/EulerHLL.h"
#include "HydroGPU/EulerEquation.h"
#include "HydroGPU/HydroGPUApp.h"
#include "Common/File.h"

EulerHLL::EulerHLL(
	HydroGPUApp &app_)
: Super(app_)
, calcEigenBasisEvent("calcEigenBasis")
, calcCFLEvent("calcCFL")
, integrateFluxEvent("integrateFlux")
{
	equation = std::make_shared<EulerEquation>(*this);
}

void EulerHLL::init() {
	Super::init();

	cl::Context context = app.context;

	entries.push_back(&calcEigenBasisEvent);
	if (!app.useFixedDT) {
		entries.push_back(&calcCFLEvent);
	}
	entries.push_back(&integrateFluxEvent);

	//memory

	int volume = app.size.s[0] * app.size.s[1] * app.size.s[2];

	eigenvaluesBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(real) * equation->numStates * volume * app.dim);
	fluxBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(real) * equation->numStates * volume * app.dim);
	
	calcFluxAndEigenvaluesKernel = cl::Kernel(program, "calcFluxAndEigenvalues");
	app.setArgs(calcFluxAndEigenvaluesKernel, eigenvaluesBuffer, fluxBuffer, stateBuffer, potentialBuffer);

	calcCFLKernel = cl::Kernel(program, "calcCFL");
	app.setArgs(calcCFLKernel, cflBuffer, eigenvaluesBuffer, app.cfl);
	
	integrateFluxKernel = cl::Kernel(program, "integrateFlux");
	app.setArgs(integrateFluxKernel, stateBuffer, fluxBuffer, dtBuffer);
}	
	
std::vector<std::string> EulerHLL::getProgramSources() {
	std::vector<std::string> sources = Super::getProgramSources();
	sources.push_back(Common::File::read("EulerHLL.cl"));
	return sources;
}

void EulerHLL::initStep() {
	//TODO calculate what's needed for the CFL here
	// for Roe solvers this is the wavespeeds (eigenvalues) 
	commands.enqueueNDRangeKernel(calcFluxAndEigenvaluesKernel, offsetNd, globalSize, localSize, NULL, &calcEigenBasisEvent.clEvent);
}

void EulerHLL::calcTimestep() {
	commands.enqueueNDRangeKernel(calcCFLKernel, offsetNd, globalSize, localSize, NULL, &calcCFLEvent.clEvent);
	findMinTimestep();	
}

void EulerHLL::step() {
	//TODO if we ever advance past Euler explicit integrators,
	// recalculate wavespeeds here
	commands.enqueueNDRangeKernel(integrateFluxKernel, offsetNd, globalSize, localSize, NULL, &integrateFluxEvent.clEvent);

	if (app.useGravity) {
		for (int i = 0; i < app.gaussSeidelMaxIter; ++i) {
			potentialBoundary();
			commands.enqueueNDRangeKernel(poissonRelaxKernel, offsetNd, globalSize, localSize);
		}
	
		commands.enqueueNDRangeKernel(addGravityKernel, offsetNd, globalSize, localSize);
		boundary();	
	}
}

