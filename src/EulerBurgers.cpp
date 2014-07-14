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
	entries.push_back(&computePressureEvent);
	entries.push_back(&diffuseMomentumEvent);
	entries.push_back(&diffuseWorkEvent);


	//memory

	int volume = getVolume();

	interfaceVelocityBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(real) * volume * app.dim);
	fluxBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(real) * numStates() * volume * app.dim);
	pressureBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(real) * volume);

	{
		//zero interface and flux
		std::vector<real> zero(volume * app.dim * numStates());
		commands.enqueueWriteBuffer(interfaceVelocityBuffer, CL_TRUE, 0, sizeof(real) * volume * app.dim, &zero[0]);
		commands.enqueueWriteBuffer(fluxBuffer, CL_TRUE, 0, sizeof(real) * numStates() * volume * app.dim, &zero[0]);
	}

	calcCFLKernel = cl::Kernel(program, "calcCFL");
	app.setArgs(calcCFLKernel, cflBuffer, stateBuffer, potentialBuffer, app.cfl);
	
	calcInterfaceVelocityKernel = cl::Kernel(program, "calcInterfaceVelocity");
	app.setArgs(calcInterfaceVelocityKernel, interfaceVelocityBuffer, stateBuffer);

	calcFluxKernel = cl::Kernel(program, "calcFlux");
	app.setArgs(calcFluxKernel, fluxBuffer, stateBuffer, interfaceVelocityBuffer, dtBuffer);

	calcFluxDerivKernel = cl::Kernel(program, "calcFluxDeriv");
	//arg0 will be provided by the integrator
	calcFluxDerivKernel.setArg(1, fluxBuffer);
	
	computePressureKernel = cl::Kernel(program, "computePressure");
	app.setArgs(computePressureKernel, pressureBuffer, stateBuffer, potentialBuffer);

	diffuseMomentumKernel = cl::Kernel(program, "diffuseMomentum");
	diffuseMomentumKernel.setArg(1, pressureBuffer);
	
	diffuseWorkKernel = cl::Kernel(program, "diffuseWork");
	diffuseWorkKernel.setArg(1, stateBuffer);
	diffuseWorkKernel.setArg(2, pressureBuffer);
}

std::vector<std::string> EulerBurgers::getProgramSources() {
	std::vector<std::string> sources = Super::getProgramSources();
	sources.push_back(Common::File::read("EulerBurgers.cl"));
	return sources;
}

void EulerBurgers::calcTimestep() {
	commands.enqueueNDRangeKernel(calcCFLKernel, offsetNd, globalSize, localSize, nullptr, &calcCFLEvent.clEvent);
	findMinTimestep();	
}

void EulerBurgers::step() {
	integrator->integrate([&](cl::Buffer derivBuffer) {
		commands.enqueueNDRangeKernel(calcInterfaceVelocityKernel, offsetNd, globalSize, localSize, nullptr, &calcInterfaceVelocityEvent.clEvent);
		commands.enqueueNDRangeKernel(calcFluxKernel, offsetNd, globalSize, localSize, nullptr, &calcFluxEvent.clEvent);
		calcFluxDerivKernel.setArg(0, derivBuffer);
		commands.enqueueNDRangeKernel(calcFluxDerivKernel, offsetNd, globalSize, localSize);
	});
	boundary();

	applyGravity();
	
	//the Hydrodynamics ii paper says it's important to diffuse momentum before work
	integrator->integrate([&](cl::Buffer derivBuffer) {
		commands.enqueueNDRangeKernel(computePressureKernel, offsetNd, globalSize, localSize, nullptr, &computePressureEvent.clEvent);
		diffuseMomentumKernel.setArg(0, derivBuffer);
		commands.enqueueNDRangeKernel(diffuseMomentumKernel, offsetNd, globalSize, localSize, nullptr, &diffuseMomentumEvent.clEvent);
	});
	boundary();
	
	integrator->integrate([&](cl::Buffer derivBuffer) {
		//commands.enqueueNDRangeKernel(computePressureKernel, offsetNd, globalSize, localSize, nullptr, &computePressureEvent.clEvent);
		diffuseWorkKernel.setArg(0, derivBuffer);
		commands.enqueueNDRangeKernel(diffuseWorkKernel, offsetNd, globalSize, localSize, nullptr, &diffuseWorkEvent.clEvent);
	});
	boundary();
}

