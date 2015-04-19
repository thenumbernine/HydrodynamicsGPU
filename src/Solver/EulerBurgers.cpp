#include "HydroGPU/Solver/EulerBurgers.h"
#include "HydroGPU/Equation/Euler.h"
#include "HydroGPU/HydroGPUApp.h"

namespace HydroGPU {
namespace Solver {

EulerBurgers::EulerBurgers(
	HydroGPUApp* app_)
: Super(app_)
, calcCFLEvent("calcCFL")
, calcInterfaceVelocityEvent("calcInterfaceVelocity")
, calcFluxEvent("calcFlux")
, computePressureEvent("computePressure")
, diffuseMomentumEvent("diffuseMomentum")
, diffuseWorkEvent("diffuseWork")
{
}

void EulerBurgers::init() {
	Super::init();
	
	cl::Context context = app->context;
	
	if (!app->useFixedDT) {
		entries.push_back(&calcCFLEvent);
	}
	entries.push_back(&calcInterfaceVelocityEvent);
	entries.push_back(&calcFluxEvent);
	entries.push_back(&computePressureEvent);
	entries.push_back(&diffuseMomentumEvent);
	entries.push_back(&diffuseWorkEvent);
}

void EulerBurgers::initBuffers() {
	Super::initBuffers();
	
	cl::Context context = app->context;
	
	int volume = getVolume();

	interfaceVelocityBuffer = clAlloc(sizeof(real) * volume * app->dim, "EulerBurgers::interfaceVelocityBuffer");

	//This is just used to add/subtract to create the derivStateCoeffBuffer one function call later.
	//I could get rid of this, but it would mean doubling up the number of calculations in the calcDerivCoeffsFromFluxCoeffs kernel.
	//real fluxStateCoeffBuffer[index:size][side:dim][l/r:2][coeff:numStates]
	fluxStateCoeffBuffer = clAlloc(sizeof(real) * numStates() * 2 * app->dim * volume, "EulerBurgers::fluxStateCoeffBuffer");

	//coefficients of the left and right neighboring state buffer to multiply and produce the interface flux
	//real derivStateCoeffBuffer[index:size][neighbor:2*dim+1][coeff:numStates]
	derivStateCoeffBuffer = clAlloc(sizeof(real) * numStates() * (1 + 2 * app->dim) * volume, "EulerBurgers::derivStateCoeffBuffer");
	pressureBuffer = clAlloc(sizeof(real) * volume, "EulerBurgers::pressureBuffer");

	//zero interface and flux
	commands.enqueueFillBuffer(interfaceVelocityBuffer, 0.f, 0, sizeof(real) * volume * app->dim);
}

void EulerBurgers::initKernels() {
	Super::initKernels();
	
	calcCFLKernel = cl::Kernel(program, "calcCFL");
	app->setArgs(calcCFLKernel, cflBuffer, stateBuffer, selfgrav->potentialBuffer, selfgrav->solidBuffer, app->cfl);
	
	calcInterfaceVelocityKernel = cl::Kernel(program, "calcInterfaceVelocity");
	app->setArgs(calcInterfaceVelocityKernel, interfaceVelocityBuffer, stateBuffer, selfgrav->solidBuffer);

	calcFluxKernel = cl::Kernel(program, "calcFlux");
	app->setArgs(calcFluxKernel, fluxStateCoeffBuffer, stateBuffer, interfaceVelocityBuffer, selfgrav->solidBuffer, dtBuffer);
	
	calcDerivCoeffsFromFluxCoeffsKernel = cl::Kernel(program, "calcDerivCoeffsFromFluxCoeffs");
	app->setArgs(calcDerivCoeffsFromFluxCoeffsKernel, derivStateCoeffBuffer, fluxStateCoeffBuffer, selfgrav->solidBuffer);

	calcDerivFromStateCoeffsKernel = cl::Kernel(program, "calcDerivFromStateCoeffs");
	//arg0 will be provided by the integrator
	calcDerivFromStateCoeffsKernel.setArg(1, stateBuffer);
	calcDerivFromStateCoeffsKernel.setArg(2, derivStateCoeffBuffer);
	calcDerivFromStateCoeffsKernel.setArg(3, selfgrav->solidBuffer);

	computePressureKernel = cl::Kernel(program, "computePressure");
	app->setArgs(computePressureKernel, pressureBuffer, stateBuffer, selfgrav->potentialBuffer, selfgrav->solidBuffer);

	diffuseMomentumKernel = cl::Kernel(program, "diffuseMomentum");
	//app->setArgs(diffuseMomentumKernel, derivStateCoeffBuffer, pressureBuffer, selfgrav->solidBuffer);
	diffuseMomentumKernel.setArg(1, pressureBuffer);
	diffuseMomentumKernel.setArg(2, selfgrav->solidBuffer);
	
	diffuseWorkKernel = cl::Kernel(program, "diffuseWork");
	diffuseWorkKernel.setArg(1, stateBuffer);
	diffuseWorkKernel.setArg(2, pressureBuffer);
	diffuseWorkKernel.setArg(3, selfgrav->solidBuffer);
}

void EulerBurgers::createEquation() {
	equation = std::make_shared<HydroGPU::Equation::Euler>(this);
}

std::vector<std::string> EulerBurgers::getProgramSources() {
	std::vector<std::string> sources = Super::getProgramSources();
	sources.push_back("#include \"EulerBurgers.cl\"\n");
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
		commands.enqueueNDRangeKernel(calcDerivCoeffsFromFluxCoeffsKernel, offsetNd, globalSize, localSize);
		
		calcDerivFromStateCoeffsKernel.setArg(0, derivBuffer);
		commands.enqueueNDRangeKernel(calcDerivFromStateCoeffsKernel, offsetNd, globalSize, localSize);
	});
	boundary();

	selfgrav->applyPotential();
	
	//the Hydrodynamics ii paper says it's important to diffuse momentum before work
	integrator->integrate([&](cl::Buffer derivBuffer) {
		commands.enqueueNDRangeKernel(computePressureKernel, offsetNd, globalSize, localSize, nullptr, &computePressureEvent.clEvent);
		
		diffuseMomentumKernel.setArg(0, derivBuffer);
		commands.enqueueNDRangeKernel(diffuseMomentumKernel, offsetNd, globalSize, localSize, nullptr, &diffuseWorkEvent.clEvent);
		
		//commands.enqueueNDRangeKernel(diffuseMomentumKernel, offsetNd, globalSize, localSize, nullptr, &diffuseMomentumEvent.clEvent);	
		//calcDerivFromStateCoeffsKernel.setArg(0, derivBuffer);
		//commands.enqueueNDRangeKernel(calcDerivFromStateCoeffsKernel, offsetNd, globalSize, localSize);
	});
	boundary();

	integrator->integrate([&](cl::Buffer derivBuffer) {
		//commands.enqueueNDRangeKernel(computePressureKernel, offsetNd, globalSize, localSize, nullptr, &computePressureEvent.clEvent);
		diffuseWorkKernel.setArg(0, derivBuffer);
		commands.enqueueNDRangeKernel(diffuseWorkKernel, offsetNd, globalSize, localSize, nullptr, &diffuseWorkEvent.clEvent);
	});
	boundary();
}

}
}

