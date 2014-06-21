#include "HydroGPU/BurgersSolver2D.h"
#include "HydroGPU/HydroGPUApp.h"

BurgersSolver2D::BurgersSolver2D(
	HydroGPUApp &app_)
: Super(app_, "Burgers2D.cl")
, calcCFLEvent("calcCFL")
, calcInterfaceVelocityEvent("calcInterfaceVelocity")
, calcFluxEvent("calcFlux")
, integrateFluxEvent("integrateFlux")
, computePressureEvent("computePressure")
, diffuseMomentumEvent("diffuseMomentum")
, diffuseWorkEvent("diffuseWork")
{
	cl::Context context = app.context;
	
	if (!app.useFixedDT) {
		entries.push_back(&calcCFLEvent);
	}
	entries.push_back(&calcInterfaceVelocityEvent);
	entries.push_back(&calcFluxEvent);
	entries.push_back(&integrateFluxEvent);
	entries.push_back(&computePressureEvent);
	entries.push_back(&diffuseMomentumEvent);
	entries.push_back(&diffuseWorkEvent);


	//memory

	int volume = app.size.s[0] * app.size.s[1] * app.size.s[2];

	interfaceVelocityBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(real) * volume * app.dim);
	fluxBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(real8) * volume * app.dim);
	pressureBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(real) * volume);

	{
		//zero interface and flux
		std::vector<real8> zero(volume * app.dim);
		commands.enqueueWriteBuffer(interfaceVelocityBuffer, CL_TRUE, 0, sizeof(real) * volume * app.dim, &zero[0]);
		commands.enqueueWriteBuffer(fluxBuffer, CL_TRUE, 0, sizeof(real8) * volume * app.dim, &zero[0]);
	}

	calcCFLKernel = cl::Kernel(program, "calcCFL");
	app.setArgs(calcCFLKernel, cflBuffer, stateBuffer, gravityPotentialBuffer, app.size, app.dx, app.cfl);
	
	calcInterfaceVelocityKernel = cl::Kernel(program, "calcInterfaceVelocity");
	app.setArgs(calcInterfaceVelocityKernel, interfaceVelocityBuffer, stateBuffer, app.size, app.dx);

	calcFluxKernel = cl::Kernel(program, "calcFlux");
	app.setArgs(calcFluxKernel, fluxBuffer, stateBuffer, interfaceVelocityBuffer, app.size, app.dx, dtBuffer);
	
	integrateFluxKernel = cl::Kernel(program, "integrateFlux");
	app.setArgs(integrateFluxKernel, stateBuffer, fluxBuffer, app.size, app.dx, dtBuffer);
	
	computePressureKernel = cl::Kernel(program, "computePressure");
	app.setArgs(computePressureKernel, pressureBuffer, stateBuffer, gravityPotentialBuffer, app.size);

	diffuseMomentumKernel = cl::Kernel(program, "diffuseMomentum");
	app.setArgs(diffuseMomentumKernel, stateBuffer, pressureBuffer, app.size, app.dx, dtBuffer);
	
	diffuseWorkKernel = cl::Kernel(program, "diffuseWork");
	app.setArgs(diffuseWorkKernel, stateBuffer, pressureBuffer, app.size, app.dx, dtBuffer);
}

void BurgersSolver2D::calcTimestep() {
	commands.enqueueNDRangeKernel(calcCFLKernel, offsetNd, globalSize, localSize, NULL, &calcCFLEvent.clEvent);
	findMinTimestep();	
}

void BurgersSolver2D::step() {
	commands.enqueueNDRangeKernel(calcInterfaceVelocityKernel, offsetNd, globalSize, localSize, NULL, &calcInterfaceVelocityEvent.clEvent);
	commands.enqueueNDRangeKernel(calcFluxKernel, offsetNd, globalSize, localSize, NULL, &calcFluxEvent.clEvent);
	commands.enqueueNDRangeKernel(integrateFluxKernel, offsetNd, globalSize, localSize, NULL, &integrateFluxEvent.clEvent);

	if (app.useGravity) {
		//recompute poisson solution to gravitational potential
		const int maxIter = 20;
		for (int i = 0; i < maxIter; ++i) {
			commands.enqueueNDRangeKernel(poissonRelaxKernel, offsetNd, globalSize, localSize);
		}
	}	

	boundary();
	
	commands.enqueueNDRangeKernel(computePressureKernel, offsetNd, globalSize, localSize, NULL, &computePressureEvent.clEvent);
	
	if (app.useGravity) {
		commands.enqueueNDRangeKernel(addGravityKernel, offsetNd, globalSize, localSize);
	
		boundary();
	}
	
	commands.enqueueNDRangeKernel(diffuseMomentumKernel, offsetNd, globalSize, localSize, NULL, &diffuseMomentumEvent.clEvent);
	
	boundary();

	commands.enqueueNDRangeKernel(diffuseWorkKernel, offsetNd, globalSize, localSize, NULL, &diffuseWorkEvent.clEvent);
}

