#include "HydroGPU/RoeSolver3D.h"
#include "HydroGPU/HydroGPUApp.h"

const int DIM = 3;

RoeSolver3D::RoeSolver3D(
	HydroGPUApp &app_)
: Super(app_, "Roe3D.cl")
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

	eigenvaluesBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(real8) * volume * DIM);
	eigenvectorsBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(real16) * volume * DIM * 2);
	eigenvectorsInverseBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(real16) * volume * DIM * 2);
	deltaQTildeBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(real8) * volume * DIM);
	fluxBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(real8) * volume * DIM);
	
	calcEigenBasisKernel = cl::Kernel(program, "calcEigenBasis");
	app.setArgs(calcEigenBasisKernel, eigenvaluesBuffer, eigenvectorsBuffer, eigenvectorsInverseBuffer, stateBuffer, gravityPotentialBuffer, app.size);

	calcCFLKernel = cl::Kernel(program, "calcCFL");
	app.setArgs(calcCFLKernel, cflBuffer, eigenvaluesBuffer, app.size, dx, app.cfl);
	
	calcDeltaQTildeKernel = cl::Kernel(program, "calcDeltaQTilde");
	app.setArgs(calcDeltaQTildeKernel, deltaQTildeBuffer, eigenvectorsInverseBuffer, stateBuffer, app.size, dx);
	
	calcFluxKernel = cl::Kernel(program, "calcFlux");
	app.setArgs(calcFluxKernel,fluxBuffer, stateBuffer, eigenvaluesBuffer, eigenvectorsBuffer, eigenvectorsInverseBuffer, deltaQTildeBuffer, app.size, dx, dtBuffer);
	
	integrateFluxKernel = cl::Kernel(program, "integrateFlux");
	app.setArgs(integrateFluxKernel, stateBuffer, fluxBuffer, app.size, dx, dtBuffer);
}	

void RoeSolver3D::initStep() {
	commands.enqueueNDRangeKernel(calcEigenBasisKernel, offset3d, globalSize, localSize, NULL, &calcEigenBasisEvent.clEvent);
}

void RoeSolver3D::calcTimestep() {
	commands.enqueueNDRangeKernel(calcCFLKernel, offset3d, globalSize, localSize, NULL, &calcCFLEvent.clEvent);
	findMinTimestep();	
}

void RoeSolver3D::step() {
	commands.enqueueNDRangeKernel(calcDeltaQTildeKernel, offset3d, globalSize, localSize, NULL, &calcDeltaQTildeEvent.clEvent);
	commands.enqueueNDRangeKernel(calcFluxKernel, offset3d, globalSize, localSize, NULL, &calcFluxEvent.clEvent);
	commands.enqueueNDRangeKernel(integrateFluxKernel, offset3d, globalSize, localSize, NULL, &integrateFluxEvent.clEvent);

	if (app.useGravity) {
		//recompute poisson solution to gravitational potential
		const int maxIter = 20;
		for (int i = 0; i < maxIter; ++i) {
			commands.enqueueNDRangeKernel(poissonRelaxKernel, offset3d, globalSize, localSize);
		}
	}	
	
	if (app.useGravity) {
		commands.enqueueNDRangeKernel(addGravityKernel, offset3d, globalSize, localSize);
		boundary();	
	}
}

