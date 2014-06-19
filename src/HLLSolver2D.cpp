#include "HydroGPU/HLLSolver2D.h"
#include "HydroGPU/HydroGPUApp.h"

HLLSolver2D::HLLSolver2D(
	HydroGPUApp &app_)
: Super(app_, "HLL2D.cl")
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

	int volume = app.size.s[0] * app.size.s[1];

	eigenvaluesBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(real4) * volume * 2);
	//deltaQTildeBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(real4) * volume * 2);
	fluxBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(real4) * volume * 2);
	
	calcEigenBasisKernel = cl::Kernel(program, "calcEigenvalues");
	app.setArgs(calcEigenBasisKernel, eigenvaluesBuffer, fluxBuffer, stateBuffer, gravityPotentialBuffer, app.size);

	calcCFLKernel = cl::Kernel(program, "calcCFL");
	app.setArgs(calcCFLKernel, cflBuffer, eigenvaluesBuffer, app.size, app.dx, app.cfl);
	
	integrateFluxKernel = cl::Kernel(program, "integrateFlux");
	app.setArgs(integrateFluxKernel, stateBuffer, fluxBuffer, app.size, app.dx, dtBuffer);
}	

void HLLSolver2D::initStep() {
	commands.enqueueNDRangeKernel(calcEigenBasisKernel, offset2d, globalSize, localSize, NULL, &calcEigenBasisEvent.clEvent);
}

void HLLSolver2D::calcTimestep() {
	commands.enqueueNDRangeKernel(calcCFLKernel, offset2d, globalSize, localSize, NULL, &calcCFLEvent.clEvent);
	findMinTimestep();	
}

void HLLSolver2D::step() {
	commands.enqueueNDRangeKernel(integrateFluxKernel, offset2d, globalSize, localSize, NULL, &integrateFluxEvent.clEvent);

	if (app.useGravity) {
		//recompute poisson solution to gravitational potential
		const int maxIter = 20;
		for (int i = 0; i < maxIter; ++i) {
			commands.enqueueNDRangeKernel(poissonRelaxKernel, offset2d, globalSize, localSize);
		}
	}	
	
	if (app.useGravity) {
		commands.enqueueNDRangeKernel(addGravityKernel, offset2d, globalSize, localSize);
		boundary();	
	}
}

