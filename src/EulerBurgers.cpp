#include "HydroGPU/EulerBurgers.h"
#include "HydroGPU/EulerEquation.h"
#include "HydroGPU/HydroGPUApp.h"
#include "Common/File.h"

EulerBurgers::EulerBurgers(
	HydroGPUApp &app_)
: Super(app_)
, calcCFLEvent("calcCFL")
, calcInterfaceVelocityEvent("calcInterfaceVelocity")
, calcFluxEvent("calcFlux")
, integrateFluxEvent("integrateFlux")
, computePressureEvent("computePressure")
, diffuseMomentumEvent("diffuseMomentum")
, diffuseWorkEvent("diffuseWork")
{
	equation = std::make_shared<EulerEquation>(*this);
}

void EulerBurgers::init() {
	Super::init();

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
	fluxBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(real) * equation->numStates * volume * app.dim);
	pressureBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(real) * volume);

	{
		//zero interface and flux
		std::vector<real> zero(volume * app.dim * equation->numStates);
		commands.enqueueWriteBuffer(interfaceVelocityBuffer, CL_TRUE, 0, sizeof(real) * volume * app.dim, &zero[0]);
		commands.enqueueWriteBuffer(fluxBuffer, CL_TRUE, 0, sizeof(real) * equation->numStates * volume * app.dim, &zero[0]);
	}

	calcCFLKernel = cl::Kernel(program, "calcCFL");
	app.setArgs(calcCFLKernel, cflBuffer, stateBuffer, gravityPotentialBuffer, app.cfl);
	
	calcInterfaceVelocityKernel = cl::Kernel(program, "calcInterfaceVelocity");
	app.setArgs(calcInterfaceVelocityKernel, interfaceVelocityBuffer, stateBuffer);

	calcFluxKernel = cl::Kernel(program, "calcFlux");
	app.setArgs(calcFluxKernel, fluxBuffer, stateBuffer, interfaceVelocityBuffer, dtBuffer);
	
	integrateFluxKernel = cl::Kernel(program, "integrateFlux");
	app.setArgs(integrateFluxKernel, stateBuffer, fluxBuffer, dtBuffer);
	
	computePressureKernel = cl::Kernel(program, "computePressure");
	app.setArgs(computePressureKernel, pressureBuffer, stateBuffer, gravityPotentialBuffer);

	diffuseMomentumKernel = cl::Kernel(program, "diffuseMomentum");
	app.setArgs(diffuseMomentumKernel, stateBuffer, pressureBuffer, dtBuffer);
	
	diffuseWorkKernel = cl::Kernel(program, "diffuseWork");
	app.setArgs(diffuseWorkKernel, stateBuffer, pressureBuffer, dtBuffer);
}

std::vector<std::string> EulerBurgers::getProgramSources() {
	std::vector<std::string> sources = Super::getProgramSources();
	sources.push_back(Common::File::read("EulerBurgers.cl"));
	return sources;
}

void EulerBurgers::calcTimestep() {
	commands.enqueueNDRangeKernel(calcCFLKernel, offsetNd, globalSize, localSize, NULL, &calcCFLEvent.clEvent);
	findMinTimestep();	
}

void EulerBurgers::step() {
	commands.enqueueNDRangeKernel(calcInterfaceVelocityKernel, offsetNd, globalSize, localSize, NULL, &calcInterfaceVelocityEvent.clEvent);
	commands.enqueueNDRangeKernel(calcFluxKernel, offsetNd, globalSize, localSize, NULL, &calcFluxEvent.clEvent);
	commands.enqueueNDRangeKernel(integrateFluxKernel, offsetNd, globalSize, localSize, NULL, &integrateFluxEvent.clEvent);

	if (app.useGravity) {
		for (int i = 0; i < app.gaussSeidelMaxIter; ++i) {
			gravityPotentialBoundary();
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

