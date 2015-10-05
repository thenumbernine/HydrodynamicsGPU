#include "HydroGPU/Solver/EulerBurgers.h"
#include "HydroGPU/Equation/Euler.h"
#include "HydroGPU/HydroGPUApp.h"

//temporary...
#include "HydroGPU/Integrator/RungeKutta4.h"

namespace HydroGPU {
namespace Solver {

EulerBurgers::EulerBurgers(
	HydroGPUApp* app_)
: Super(app_)
{
}

void EulerBurgers::initBuffers() {
	Super::initBuffers();
	
	int volume = getVolume();

	interfaceVelocityBuffer = clAlloc(sizeof(real) * volume, "EulerBurgers::interfaceVelocityBuffer");
	fluxBuffer = clAlloc(sizeof(real) * numStates() * volume, "EulerBurgers::fluxBuffer");
	pressureBuffer = clAlloc(sizeof(real) * volume, "EulerBurgers::pressureBuffer");

	//zero interface and flux
	commands.enqueueFillBuffer(interfaceVelocityBuffer, 0.f, 0, sizeof(real) * volume);
}

void EulerBurgers::initKernels() {
	Super::initKernels();
	
	findMinTimestepKernel = cl::Kernel(program, "findMinTimestep");
	CLCommon::setArgs(findMinTimestepKernel, dtBuffer, stateBuffer, selfgrav->potentialBuffer, selfgrav->solidBuffer);
	
	calcInterfaceVelocityKernel = cl::Kernel(program, "calcInterfaceVelocity");
	CLCommon::setArgs(calcInterfaceVelocityKernel, interfaceVelocityBuffer, stateBuffer, selfgrav->solidBuffer);
	
	calcFluxKernel = cl::Kernel(program, "calcFlux");
	CLCommon::setArgs(calcFluxKernel, fluxBuffer, stateBuffer, interfaceVelocityBuffer, selfgrav->solidBuffer);
	
	calcFluxDerivKernel = cl::Kernel(program, "calcFluxDeriv");
	calcFluxDerivKernel.setArg(1, fluxBuffer);
	calcFluxDerivKernel.setArg(3, selfgrav->solidBuffer);
	
	computePressureKernel = cl::Kernel(program, "computePressure");
	CLCommon::setArgs(computePressureKernel, pressureBuffer, stateBuffer, selfgrav->potentialBuffer, selfgrav->solidBuffer);
	
	diffuseMomentumKernel = cl::Kernel(program, "diffuseMomentum");
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
	sources.insert(sources.begin(), "#define SOLID 1\n");
	sources.push_back("#include \"EulerBurgers.cl\"\n");
	return sources;
}

real EulerBurgers::calcTimestep() {
	commands.enqueueNDRangeKernel(findMinTimestepKernel, offsetNd, globalSize, localSize);

	return findMinTimestep();
}

void EulerBurgers::step(real dt) {
	int sideStart, sideEnd, sideStep;
	getSideRange(sideStart, sideEnd, sideStep);
	for (int side = sideStart; side != sideEnd; side += sideStep) {
		integrator->integrate(dt, [&](cl::Buffer derivBuffer) {
			calcInterfaceVelocityKernel.setArg(3, side);
			commands.enqueueNDRangeKernel(calcInterfaceVelocityKernel, offsetNd, globalSize, localSize);
			
			calcFluxKernel.setArg(4, dt);
			calcFluxKernel.setArg(5, side);
			commands.enqueueNDRangeKernel(calcFluxKernel, offsetNd, globalSize, localSize);

			calcFluxDerivKernel.setArg(0, derivBuffer);
			calcFluxDerivKernel.setArg(2, side);
			commands.enqueueNDRangeKernel(calcFluxDerivKernel, offsetNd, globalSize, localSize);
		});
	}
	boundary();

	selfgrav->applyPotential(dt);
	
	//the Hydrodynamics ii paper says it's important to diffuse momentum before work
	integrator->integrate(dt, [&](cl::Buffer derivBuffer) {
		
		commands.enqueueNDRangeKernel(computePressureKernel, offsetNd, globalSize, localSize);

		diffuseMomentumKernel.setArg(0, derivBuffer);
		commands.enqueueNDRangeKernel(diffuseMomentumKernel, offsetNd, globalSize, localSize);
	});
	boundary();

	integrator->integrate(dt, [&](cl::Buffer derivBuffer) {
		//computePressureFunc(pressureBuffer, stateBuffer, selfgrav->potentialBuffer, selfgrav->solidBuffer);
		diffuseWorkKernel.setArg(0, derivBuffer);
		commands.enqueueNDRangeKernel(diffuseWorkKernel, offsetNd, globalSize, localSize);
	});
	boundary();
}

}
}

