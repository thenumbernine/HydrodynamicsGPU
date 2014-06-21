#include "HydroGPU/RoeSolver2D.h"
#include "HydroGPU/HydroGPUApp.h"

RoeSolver2D::RoeSolver2D(
	HydroGPUApp &app_)
: Super(app_, "Roe2D.cl")
, calcEigenBasisEvent("calcEigenBasis")
, calcCFLEvent("calcCFL")
, calcDeltaQTildeEvent("calcDeltaQTilde")
, calcFluxEvent("calcFlux")
, integrateFluxEvent("integrateFlux")
{
	cl::Context context = app.context;

	entries.push_back(&calcEigenBasisEvent);
	if (!app.useFixedDT) {
		entries.push_back(&calcCFLEvent);
	}
	entries.push_back(&calcDeltaQTildeEvent);
	entries.push_back(&calcFluxEvent);
	entries.push_back(&integrateFluxEvent);

	//memory

	int volume = app.size.s[0] * app.size.s[1] * app.size.s[2];

	eigenvaluesBuffer = clAlloc(sizeof(real8) * volume * app.dim);
	eigenvectorsBuffer = clAlloc(sizeof(real16) * 4 * volume * app.dim);
	eigenvectorsInverseBuffer = clAlloc(sizeof(real16) * 4 * volume * app.dim);
	deltaQTildeBuffer = clAlloc(sizeof(real8) * volume * app.dim);
	fluxBuffer = clAlloc(sizeof(real8) * volume * app.dim);
	
	calcEigenBasisKernel = cl::Kernel(program, "calcEigenBasis");
	app.setArgs(calcEigenBasisKernel, eigenvaluesBuffer, eigenvectorsBuffer, eigenvectorsInverseBuffer, stateBuffer, gravityPotentialBuffer);

	calcCFLKernel = cl::Kernel(program, "calcCFL");
	app.setArgs(calcCFLKernel, cflBuffer, eigenvaluesBuffer, app.dx, app.cfl);
	
	calcDeltaQTildeKernel = cl::Kernel(program, "calcDeltaQTilde");
	app.setArgs(calcDeltaQTildeKernel, deltaQTildeBuffer, eigenvectorsInverseBuffer, stateBuffer, app.dx);
	
	calcFluxKernel = cl::Kernel(program, "calcFlux");
	//cl.hpp...
	//app.setArgs(calcFluxKernel,fluxBuffer, stateBuffer, eigenvaluesBuffer, eigenvectorsBuffer, eigenvectorsInverseBuffer, deltaQTildeBuffer, app.dx, dtBuffer);
	//just in case it is using a sizeof() or something and CL isn't bounds checking the memory being written to (esp when writing app.dx which is a real3)
	clSetKernelArg(calcFluxKernel(), 0, sizeof(cl_mem), &fluxBuffer());
	clSetKernelArg(calcFluxKernel(), 1, sizeof(cl_mem), &stateBuffer());
	clSetKernelArg(calcFluxKernel(), 2, sizeof(cl_mem), &eigenvaluesBuffer());
	clSetKernelArg(calcFluxKernel(), 3, sizeof(cl_mem), &eigenvectorsBuffer());
	clSetKernelArg(calcFluxKernel(), 4, sizeof(cl_mem), &eigenvectorsInverseBuffer());
	clSetKernelArg(calcFluxKernel(), 5, sizeof(cl_mem), &deltaQTildeBuffer());
	clSetKernelArg(calcFluxKernel(), 6, sizeof(real4), &app.dx);
	clSetKernelArg(calcFluxKernel(), 7, sizeof(cl_mem), &dtBuffer());
	
	integrateFluxKernel = cl::Kernel(program, "integrateFlux");
	app.setArgs(integrateFluxKernel, stateBuffer, fluxBuffer, app.size, app.dx, dtBuffer);
}	

void RoeSolver2D::initStep() {
	commands.enqueueNDRangeKernel(calcEigenBasisKernel, offsetNd, globalSize, localSize, NULL, &calcEigenBasisEvent.clEvent);
}

void RoeSolver2D::calcTimestep() {
	commands.enqueueNDRangeKernel(calcCFLKernel, offsetNd, globalSize, localSize, NULL, &calcCFLEvent.clEvent);
	findMinTimestep();	
}

void RoeSolver2D::step() {
	commands.enqueueNDRangeKernel(calcDeltaQTildeKernel, offsetNd, globalSize, localSize, NULL, &calcDeltaQTildeEvent.clEvent);
	commands.enqueueNDRangeKernel(calcFluxKernel, offsetNd, globalSize, localSize, NULL, &calcFluxEvent.clEvent);
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

