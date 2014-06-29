#include "HydroGPU/EulerHLL.h"
#include "HydroGPU/HydroGPUApp.h"
#include "Common/File.h"

EulerHLL::EulerHLL(
	HydroGPUApp &app_)
: Super(app_)
, calcEigenBasisEvent("calcEigenBasis")
, calcCFLEvent("calcCFL")
, integrateFluxEvent("integrateFlux")
{
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

	eigenvaluesBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(real) * numStates * volume * app.dim);
	fluxBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(real) * numStates * volume * app.dim);
	
	calcFluxAndEigenvaluesKernel = cl::Kernel(program, "calcFluxAndEigenvalues");
	app.setArgs(calcFluxAndEigenvaluesKernel, eigenvaluesBuffer, fluxBuffer, stateBuffer, gravityPotentialBuffer);

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
	commands.enqueueNDRangeKernel(calcFluxAndEigenvaluesKernel, offsetNd, globalSize, localSize, NULL, &calcEigenBasisEvent.clEvent);
}

void EulerHLL::calcTimestep() {
	commands.enqueueNDRangeKernel(calcCFLKernel, offsetNd, globalSize, localSize, NULL, &calcCFLEvent.clEvent);
	findMinTimestep();	
}

void EulerHLL::step() {
	commands.enqueueNDRangeKernel(integrateFluxKernel, offsetNd, globalSize, localSize, NULL, &integrateFluxEvent.clEvent);

	if (app.useGravity) {
		//recompute poisson solution to gravitational potential
		const int maxIter = 20;
		for (int i = 0; i < maxIter; ++i) {
			commands.enqueueNDRangeKernel(poissonRelaxKernel, offsetNd, globalSize, localSize);
		}
	}	
	
	if (app.useGravity) {
		commands.enqueueNDRangeKernel(addGravityKernel, offsetNd, globalSize, localSize);
		boundary();	
	}
}

