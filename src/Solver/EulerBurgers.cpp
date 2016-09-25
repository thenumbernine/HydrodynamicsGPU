#include "HydroGPU/Solver/EulerBurgers.h"
#include "HydroGPU/Equation/Euler.h"
#include "HydroGPU/HydroGPUApp.h"

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

	interfaceVelocityBuffer = cl.alloc(sizeof(real) * volume * app->dim, "EulerBurgers::interfaceVelocityBuffer");
	pressureBuffer = cl.alloc(sizeof(real) * volume, "EulerBurgers::pressureBuffer");

	cl.zero(interfaceVelocityBuffer, volume * app->dim * sizeof(real));
}

void EulerBurgers::initKernels() {
	Super::initKernels();
	
	calcCellTimestepKernel = cl::Kernel(program, "calcCellTimestep");
	CLCommon::setArgs(calcCellTimestepKernel, dtBuffer, stateBuffer, selfgrav->potentialBuffer, selfgrav->solidBuffer);
	
	calcInterfaceVelocityKernel = cl::Kernel(program, "calcInterfaceVelocity");
	CLCommon::setArgs(calcInterfaceVelocityKernel, interfaceVelocityBuffer, stateBuffer, selfgrav->solidBuffer);
	
	calcFluxKernel.setArg(2, interfaceVelocityBuffer);
	calcFluxKernel.setArg(3, selfgrav->solidBuffer);
	
	calcFluxDerivKernel.setArg(2, selfgrav->solidBuffer);
	
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
	equation = std::make_shared<HydroGPU::Equation::Euler>(app);
}

std::vector<std::string> EulerBurgers::getProgramSources() {
	std::vector<std::string> sources = Super::getProgramSources();
	sources.insert(sources.begin(), "#define SOLID 1\n");
	sources.push_back("#include \"EulerBurgers.cl\"\n");
	return sources;
}

real EulerBurgers::calcTimestep() {
	commands.enqueueNDRangeKernel(calcCellTimestepKernel, offsetNd, globalSize, localSize);

	return findMinTimestep();
}

void EulerBurgers::step(real dt) {
	integrator->integrate(dt, [&](cl::Buffer derivBuffer) {
		commands.enqueueNDRangeKernel(calcInterfaceVelocityKernel, offsetNd, globalSize, localSize);
		
		calcFluxKernel.setArg(4, dt);
		commands.enqueueNDRangeKernel(calcFluxKernel, offsetNd, globalSize, localSize);

		calcFluxDerivKernel.setArg(0, derivBuffer);
		commands.enqueueNDRangeKernel(calcFluxDerivKernel, offsetNd, globalSize, localSize);
	});
	
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
