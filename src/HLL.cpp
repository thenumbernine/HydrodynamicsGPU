#include "HydroGPU/HLL.h"
#include "HydroGPU/HydroGPUApp.h"

HLL::HLL(
	HydroGPUApp &app_)
: Super(app_, "HLL.cl")
, calcEigenBasisEvent("calcEigenBasis")
, calcCFLEvent("calcCFL")
, integrateFluxEvent("integrateFlux")
{
	cl::Context context = app.context;

	entries.push_back(&calcEigenBasisEvent);
	if (!app.useFixedDT) {
		entries.push_back(&calcCFLEvent);
	}
	entries.push_back(&integrateFluxEvent);

	//memory

	int volume = app.size.s[0] * app.size.s[1] * app.size.s[2];

	eigenvaluesBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(real8) * volume * app.dim);
	fluxBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(real8) * volume * app.dim);
	
	calcFluxAndEigenvaluesKernel = cl::Kernel(program, "calcFluxAndEigenvalues");
	app.setArgs(calcFluxAndEigenvaluesKernel, eigenvaluesBuffer, fluxBuffer, stateBuffer, gravityPotentialBuffer);

	calcCFLKernel = cl::Kernel(program, "calcCFL");
	app.setArgs(calcCFLKernel, cflBuffer, eigenvaluesBuffer, app.cfl);
	
	integrateFluxKernel = cl::Kernel(program, "integrateFlux");
	app.setArgs(integrateFluxKernel, stateBuffer, fluxBuffer, dtBuffer);
}	

void HLL::initStep() {
	commands.enqueueNDRangeKernel(calcFluxAndEigenvaluesKernel, offsetNd, globalSize, localSize, NULL, &calcEigenBasisEvent.clEvent);
}

void HLL::calcTimestep() {
	commands.enqueueNDRangeKernel(calcCFLKernel, offsetNd, globalSize, localSize, NULL, &calcCFLEvent.clEvent);
	findMinTimestep();	
}

void HLL::step() {
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

