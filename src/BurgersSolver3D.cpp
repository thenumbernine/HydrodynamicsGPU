#include "HydroGPU/BurgersSolver3D.h"
#include "HydroGPU/HydroGPUApp.h"

const int DIM = 3;

BurgersSolver3D::BurgersSolver3D(
	HydroGPUApp &app_)
: Super(app_, "Burgers3D.cl")
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

	interfaceVelocityBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(real) * volume * DIM);
	fluxBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(real8) * volume * DIM);
	pressureBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(real) * volume);

	{
		//zero interface and flux
		std::vector<real8> zero(volume * DIM);
		commands.enqueueWriteBuffer(interfaceVelocityBuffer, CL_TRUE, 0, sizeof(real) * volume * DIM, &zero[0]);
		commands.enqueueWriteBuffer(fluxBuffer, CL_TRUE, 0, sizeof(real8) * volume * DIM, &zero[0]);
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

void BurgersSolver3D::calcTimestep() {
	commands.enqueueNDRangeKernel(calcCFLKernel, offset3d, globalSize, localSize, NULL, &calcCFLEvent.clEvent);
	findMinTimestep();	
}

void BurgersSolver3D::step() {
	commands.enqueueNDRangeKernel(calcInterfaceVelocityKernel, offset3d, globalSize, localSize, NULL, &calcInterfaceVelocityEvent.clEvent);
	commands.enqueueNDRangeKernel(calcFluxKernel, offset3d, globalSize, localSize, NULL, &calcFluxEvent.clEvent);
	commands.enqueueNDRangeKernel(integrateFluxKernel, offset3d, globalSize, localSize, NULL, &integrateFluxEvent.clEvent);

	if (app.useGravity) {
		//recompute poisson solution to gravitational potential
		const int maxIter = 20;
		for (int i = 0; i < maxIter; ++i) {
			commands.enqueueNDRangeKernel(poissonRelaxKernel, offset3d, globalSize, localSize);
		}
	}	

	boundary();
	
	commands.enqueueNDRangeKernel(computePressureKernel, offset3d, globalSize, localSize, NULL, &computePressureEvent.clEvent);
	
	if (app.useGravity) {
		commands.enqueueNDRangeKernel(addGravityKernel, offset3d, globalSize, localSize);
	
		boundary();
	}
	
	commands.enqueueNDRangeKernel(diffuseMomentumKernel, offset3d, globalSize, localSize, NULL, &diffuseMomentumEvent.clEvent);
	
	boundary();

	commands.enqueueNDRangeKernel(diffuseWorkKernel, offset3d, globalSize, localSize, NULL, &diffuseWorkEvent.clEvent);
}

