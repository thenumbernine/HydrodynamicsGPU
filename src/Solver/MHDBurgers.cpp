#include "HydroGPU/Solver/MHDBurgers.h"
#include "HydroGPU/Equation/MHD.h"
#include "HydroGPU/HydroGPUApp.h"

namespace HydroGPU {
namespace Solver {

void MHDBurgers::init() {
	Super::init();

	cl::Context context = app->context;

	//memory

	int volume = getVolume();

	interfaceVelocityBuffer = clAlloc(sizeof(real) * volume * app->dim);
	interfaceMagneticFieldBuffer = clAlloc(sizeof(real) * volume * app->dim);
	fluxBuffer = clAlloc(sizeof(real) * numStates() * volume * app->dim);
	pressureBuffer = clAlloc(sizeof(real) * volume);

	commands.enqueueFillBuffer(interfaceVelocityBuffer, 0.f, 0, sizeof(real) * volume * app->dim);
	commands.enqueueFillBuffer(interfaceMagneticFieldBuffer, 0.f, 0, sizeof(real) * volume * app->dim);
	commands.enqueueFillBuffer(fluxBuffer, 0.f, 0, sizeof(real) * numStates() * volume * app->dim);

	calcCFLKernel = cl::Kernel(program, "calcCFL");
	app->setArgs(calcCFLKernel, cflBuffer, stateBuffer, selfgrav->potentialBuffer, app->cfl);
	
	calcInterfaceVelocityKernel = cl::Kernel(program, "calcInterfaceVelocity");
	app->setArgs(calcInterfaceVelocityKernel, interfaceVelocityBuffer, stateBuffer);
	
	calcInterfaceMagneticFieldKernel = cl::Kernel(program, "calcInterfaceMagneticField");
	app->setArgs(calcInterfaceMagneticFieldKernel, interfaceMagneticFieldBuffer, stateBuffer);

	calcVelocityFluxKernel = cl::Kernel(program, "calcVelocityFlux");
	app->setArgs(calcVelocityFluxKernel, fluxBuffer, stateBuffer, interfaceVelocityBuffer, dtBuffer);

	calcMagneticFieldFluxKernel = cl::Kernel(program, "calcMagneticFieldFlux");
	app->setArgs(calcMagneticFieldFluxKernel, fluxBuffer, stateBuffer, interfaceMagneticFieldBuffer, dtBuffer);

	calcFluxDerivKernel = cl::Kernel(program, "calcFluxDeriv");
	//arg0 will be provided by the integrator
	calcFluxDerivKernel.setArg(1, fluxBuffer);
	
	computePressureKernel = cl::Kernel(program, "computePressure");
	app->setArgs(computePressureKernel, pressureBuffer, stateBuffer, selfgrav->potentialBuffer);

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

void MHDBurgers::calcTimestep() {
	commands.enqueueNDRangeKernel(calcCFLKernel, offsetNd, globalSize, localSize);
	findMinTimestep();	
}

void MHDBurgers::step() {
	advectVelocity();	
	divfree->update();
	advectMagneticField();
	divfree->update();
	selfgrav->applyPotential();
	diffusePressure();
	diffuseWork();
	divfree->update();
}

void MHDBurgers::advectVelocity() {
	integrator->integrate([&](cl::Buffer derivBuffer) {
		commands.enqueueNDRangeKernel(calcInterfaceVelocityKernel, offsetNd, globalSize, localSize);
		commands.enqueueNDRangeKernel(calcVelocityFluxKernel, offsetNd, globalSize, localSize);
		calcFluxDerivKernel.setArg(0, derivBuffer);
		commands.enqueueNDRangeKernel(calcFluxDerivKernel, offsetNd, globalSize, localSize);
	});
	boundary();
}

void MHDBurgers::advectMagneticField() {
	integrator->integrate([&](cl::Buffer derivBuffer) {
		commands.enqueueNDRangeKernel(calcInterfaceMagneticFieldKernel, offsetNd, globalSize, localSize);
		commands.enqueueNDRangeKernel(calcMagneticFieldFluxKernel, offsetNd, globalSize, localSize);
		calcFluxDerivKernel.setArg(0, derivBuffer);
		commands.enqueueNDRangeKernel(calcFluxDerivKernel, offsetNd, globalSize, localSize);
	});
	boundary();
}

void MHDBurgers::diffusePressure() {
	//the Hydrodynamics ii paper says it's important to diffuse momentum before work
	integrator->integrate([&](cl::Buffer derivBuffer) {
		commands.enqueueNDRangeKernel(computePressureKernel, offsetNd, globalSize, localSize);
		diffuseMomentumKernel.setArg(0, derivBuffer);
		commands.enqueueNDRangeKernel(diffuseMomentumKernel, offsetNd, globalSize, localSize);
	});
	boundary();
}

void MHDBurgers::diffuseWork() {
	integrator->integrate([&](cl::Buffer derivBuffer) {
		//commands.enqueueNDRangeKernel(computePressureKernel, offsetNd, globalSize, localSize);
		diffuseWorkKernel.setArg(0, derivBuffer);
		commands.enqueueNDRangeKernel(diffuseWorkKernel, offsetNd, globalSize, localSize);
	});
	boundary();
}

}
}

