#include "HydroGPU/Solver/MHDBurgers.h"
#include "HydroGPU/Equation/MHD.h"
#include "HydroGPU/HydroGPUApp.h"

namespace HydroGPU {
namespace Solver {

void MHDBurgers::initBuffers() {
	Super::initBuffers();

	int volume = getVolume();

	interfaceVelocityBuffer = cl.alloc(sizeof(real) * volume * app->dim);
	interfaceMagneticFieldBuffer = cl.alloc(sizeof(real) * volume * app->dim);
	pressureBuffer = cl.alloc(sizeof(real) * volume);

	cl.zero(interfaceVelocityBuffer, volume * app->dim);
	cl.zero(interfaceMagneticFieldBuffer, volume * app->dim);
}

void MHDBurgers::initKernels() {
	Super::initKernels();

	calcCellTimestepKernel = cl::Kernel(program, "calcCellTimestep");
	CLCommon::setArgs(calcCellTimestepKernel, dtBuffer, stateBuffer, selfgrav->potentialBuffer);
	
	calcInterfaceVelocityKernel = cl::Kernel(program, "calcInterfaceVelocity");
	CLCommon::setArgs(calcInterfaceVelocityKernel, interfaceVelocityBuffer, stateBuffer);
	
	calcInterfaceMagneticFieldKernel = cl::Kernel(program, "calcInterfaceMagneticField");
	CLCommon::setArgs(calcInterfaceMagneticFieldKernel, interfaceMagneticFieldBuffer, stateBuffer);

	calcVelocityFluxKernel = cl::Kernel(program, "calcVelocityFlux");
	CLCommon::setArgs(calcVelocityFluxKernel, fluxBuffer, stateBuffer, interfaceVelocityBuffer);

	calcMagneticFieldFluxKernel = cl::Kernel(program, "calcMagneticFieldFlux");
	CLCommon::setArgs(calcMagneticFieldFluxKernel, fluxBuffer, stateBuffer, interfaceMagneticFieldBuffer);

	computePressureKernel = cl::Kernel(program, "computePressure");
	CLCommon::setArgs(computePressureKernel, pressureBuffer, stateBuffer, selfgrav->potentialBuffer);

	diffuseMomentumKernel = cl::Kernel(program, "diffuseMomentum");
	diffuseMomentumKernel.setArg(1, pressureBuffer);
	
	diffuseWorkKernel = cl::Kernel(program, "diffuseWork");
	diffuseWorkKernel.setArg(1, stateBuffer);
	diffuseWorkKernel.setArg(2, pressureBuffer);
}

void MHDBurgers::createEquation() {
	equation = std::make_shared<HydroGPU::Equation::MHD>(this);
}

std::vector<std::string> MHDBurgers::getProgramSources() {
	std::vector<std::string> sources = Super::getProgramSources();
	sources.push_back("#include \"MHDBurgers.cl\"\n");
	return sources;
}

real MHDBurgers::calcTimestep() {
	commands.enqueueNDRangeKernel(calcCellTimestepKernel, offsetNd, globalSize, localSize);
	return findMinTimestep();	
}

void MHDBurgers::step(real dt) {
	advectVelocity(dt);
	divfree->update();
	advectMagneticField(dt);
	divfree->update();
	selfgrav->applyPotential(dt);
	diffusePressure(dt);
	diffuseWork(dt);
	divfree->update();
}

void MHDBurgers::advectVelocity(real dt) {
	calcVelocityFluxKernel.setArg(3, dt);
	integrator->integrate(dt, [&](cl::Buffer derivBuffer) {
		commands.enqueueNDRangeKernel(calcInterfaceVelocityKernel, offsetNd, globalSize, localSize);
		commands.enqueueNDRangeKernel(calcVelocityFluxKernel, offsetNd, globalSize, localSize);
		calcFluxDerivKernel.setArg(0, derivBuffer);
		commands.enqueueNDRangeKernel(calcFluxDerivKernel, offsetNd, globalSize, localSize);
	});
	boundary();
}

void MHDBurgers::advectMagneticField(real dt) {
	calcMagneticFieldFluxKernel.setArg(3, dt);
	integrator->integrate(dt, [&](cl::Buffer derivBuffer) {
		commands.enqueueNDRangeKernel(calcInterfaceMagneticFieldKernel, offsetNd, globalSize, localSize);
		commands.enqueueNDRangeKernel(calcMagneticFieldFluxKernel, offsetNd, globalSize, localSize);
		calcFluxDerivKernel.setArg(0, derivBuffer);
		commands.enqueueNDRangeKernel(calcFluxDerivKernel, offsetNd, globalSize, localSize);
	});
	boundary();
}

void MHDBurgers::diffusePressure(real dt) {
	//the Hydrodynamics ii paper says it's important to diffuse momentum before work
	integrator->integrate(dt, [&](cl::Buffer derivBuffer) {
		commands.enqueueNDRangeKernel(computePressureKernel, offsetNd, globalSize, localSize);
		diffuseMomentumKernel.setArg(0, derivBuffer);
		commands.enqueueNDRangeKernel(diffuseMomentumKernel, offsetNd, globalSize, localSize);
	});
	boundary();
}

void MHDBurgers::diffuseWork(real dt) {
	integrator->integrate(dt, [&](cl::Buffer derivBuffer) {
		//commands.enqueueNDRangeKernel(computePressureKernel, offsetNd, globalSize, localSize);
		diffuseWorkKernel.setArg(0, derivBuffer);
		commands.enqueueNDRangeKernel(diffuseWorkKernel, offsetNd, globalSize, localSize);
	});
	boundary();
}

}
}

