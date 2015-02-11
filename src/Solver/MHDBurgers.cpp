#include "HydroGPU/Solver/MHDBurgers.h"
#include "HydroGPU/Equation/MHD.h"
#include "HydroGPU/HydroGPUApp.h"
#include "Common/File.h"

namespace HydroGPU {
namespace Solver {

MHDBurgers::MHDBurgers(HydroGPUApp& app_)
: Super(app_)
{
	equation = std::make_shared<HydroGPU::Equation::MHD>(*this);
}

void MHDBurgers::init() {
	Super::init();

	cl::Context context = app.context;

	//memory

	int volume = getVolume();

	interfaceVelocityBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(real) * volume * app.dim);
	interfaceMagneticFieldBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(real) * volume * app.dim);
	fluxBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(real) * numStates() * volume * app.dim);
	pressureBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(real) * volume);
	magneticFieldDivergenceBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(real) * volume);
	magneticFieldPotentialBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(real) * volume);

	commands.enqueueFillBuffer(interfaceVelocityBuffer, 0.f, 0, sizeof(real) * volume * app.dim);
	commands.enqueueFillBuffer(interfaceMagneticFieldBuffer, 0.f, 0, sizeof(real) * volume * app.dim);
	commands.enqueueFillBuffer(fluxBuffer, 0.f, 0, sizeof(real) * numStates() * volume * app.dim);

	calcCFLKernel = cl::Kernel(program, "calcCFL");
	app.setArgs(calcCFLKernel, cflBuffer, stateBuffer, potentialBuffer, app.cfl);
	
	calcInterfaceVelocityKernel = cl::Kernel(program, "calcInterfaceVelocity");
	app.setArgs(calcInterfaceVelocityKernel, interfaceVelocityBuffer, stateBuffer);
	
	calcInterfaceMagneticFieldKernel = cl::Kernel(program, "calcInterfaceMagneticField");
	app.setArgs(calcInterfaceMagneticFieldKernel, interfaceMagneticFieldBuffer, stateBuffer);

	calcVelocityFluxKernel = cl::Kernel(program, "calcVelocityFlux");
	app.setArgs(calcVelocityFluxKernel, fluxBuffer, stateBuffer, interfaceVelocityBuffer, dtBuffer);

	calcMagneticFieldFluxKernel = cl::Kernel(program, "calcMagneticFieldFlux");
	app.setArgs(calcMagneticFieldFluxKernel, fluxBuffer, stateBuffer, interfaceMagneticFieldBuffer, dtBuffer);

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

	calcMagneticFieldDivergenceKernel = cl::Kernel(program, "calcMagneticFieldDivergence");
	app.setArgs(calcMagneticFieldDivergenceKernel, magneticFieldDivergenceBuffer, stateBuffer);

	magneticPotentialPoissonRelaxKernel = cl::Kernel(program, "magneticPotentialPoissonRelax");
	app.setArgs(magneticPotentialPoissonRelaxKernel, magneticFieldPotentialBuffer, magneticFieldDivergenceBuffer);
	
	magneticFieldRemoveDivergenceKernel = cl::Kernel(program, "magneticFieldRemoveDivergence");
	app.setArgs(magneticFieldRemoveDivergenceKernel, stateBuffer, magneticFieldPotentialBuffer);
}

std::vector<std::string> MHDBurgers::getProgramSources() {
	std::vector<std::string> sources = Super::getProgramSources();
	sources.push_back(Common::File::read("MHDBurgers.cl"));
	return sources;
}

void MHDBurgers::calcTimestep() {
	commands.enqueueNDRangeKernel(calcCFLKernel, offsetNd, globalSize, localSize);
	findMinTimestep();	
}

void MHDBurgers::step() {
	integrator->integrate([&](cl::Buffer derivBuffer) {
		commands.enqueueNDRangeKernel(calcInterfaceVelocityKernel, offsetNd, globalSize, localSize);
		commands.enqueueNDRangeKernel(calcVelocityFluxKernel, offsetNd, globalSize, localSize);
		calcFluxDerivKernel.setArg(0, derivBuffer);
		commands.enqueueNDRangeKernel(calcFluxDerivKernel, offsetNd, globalSize, localSize);
	});
	boundary();

	integrator->integrate([&](cl::Buffer derivBuffer) {
		commands.enqueueNDRangeKernel(calcInterfaceMagneticFieldKernel, offsetNd, globalSize, localSize);
		commands.enqueueNDRangeKernel(calcMagneticFieldFluxKernel, offsetNd, globalSize, localSize);
		calcFluxDerivKernel.setArg(0, derivBuffer);
		commands.enqueueNDRangeKernel(calcFluxDerivKernel, offsetNd, globalSize, localSize);
	});
	boundary();

	applyPotential();
	
	//the Hydrodynamics ii paper says it's important to diffuse momentum before work
	integrator->integrate([&](cl::Buffer derivBuffer) {
		commands.enqueueNDRangeKernel(computePressureKernel, offsetNd, globalSize, localSize);
		diffuseMomentumKernel.setArg(0, derivBuffer);
		commands.enqueueNDRangeKernel(diffuseMomentumKernel, offsetNd, globalSize, localSize);
	});
	boundary();
	
	integrator->integrate([&](cl::Buffer derivBuffer) {
		//commands.enqueueNDRangeKernel(computePressureKernel, offsetNd, globalSize, localSize);
		diffuseWorkKernel.setArg(0, derivBuffer);
		commands.enqueueNDRangeKernel(diffuseWorkKernel, offsetNd, globalSize, localSize);
	});
	boundary();

	//calculate divergence
	commands.enqueueNDRangeKernel(calcMagneticFieldDivergenceKernel, offsetNd, globalSize, localSize);

	//poisson relax divergence into potential buffer
	int volume = getVolume();
	commands.enqueueFillBuffer(magneticFieldPotentialBuffer, 0.f, 0, sizeof(real) * volume);
	for (int i = 0; i < app.gaussSeidelMaxIter; ++i) {
		magneticFieldPotentialBoundary();	//boundary to magnetic field potential buffer
		commands.enqueueNDRangeKernel(magneticPotentialPoissonRelaxKernel, offsetNd, globalSize, localSize);
	}
	commands.enqueueNDRangeKernel(magneticFieldRemoveDivergenceKernel, offsetNd, globalSize, localSize);
}

//looks a lot like Solver::potentialBoundary
// maybe I could merge them?
void MHDBurgers::magneticFieldPotentialBoundary() {
	cl::NDRange offset, global, local;
	for (int i = 0; i < app.dim; ++i) {
		int boundaryKernelIndex = equation->gravityGetBoundaryKernelForBoundaryMethod(*this, i);
		cl::Kernel& kernel = boundaryKernels[boundaryKernelIndex][i];
		app.setArgs(kernel, magneticFieldPotentialBuffer, 1, 0);
		getBoundaryRanges(i, offset, global, local);
		commands.enqueueNDRangeKernel(kernel, offset, global, local);
	}
}

}
}

