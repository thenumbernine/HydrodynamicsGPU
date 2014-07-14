#include "HydroGPU/EulerHLL.h"
#include "HydroGPU/EulerEquation.h"
#include "HydroGPU/HydroGPUApp.h"
#include "Common/File.h"

EulerHLL::EulerHLL(
	HydroGPUApp &app_)
: Super(app_)
{
	equation = std::make_shared<EulerEquation>(*this);
}

void EulerHLL::init() {
	Super::init();

	cl::Context context = app.context;

	//memory

	int volume = getVolume();

	eigenvaluesBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(real) * numStates() * volume * app.dim);
	fluxBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(real) * numStates() * volume * app.dim);
	
	calcEigenvaluesKernel = cl::Kernel(program, "calcEigenvalues");
	app.setArgs(calcEigenvaluesKernel, eigenvaluesBuffer, stateBuffer, potentialBuffer);
	
	calcFluxKernel = cl::Kernel(program, "calcFlux");
	app.setArgs(calcFluxKernel, fluxBuffer, stateBuffer, potentialBuffer);

	calcCFLKernel = cl::Kernel(program, "calcCFL");
	app.setArgs(calcCFLKernel, cflBuffer, eigenvaluesBuffer, app.cfl);
	
	calcFluxDerivKernel = cl::Kernel(program, "calcFluxDeriv");
	calcFluxDerivKernel.setArg(1, fluxBuffer);
}	
	
std::vector<std::string> EulerHLL::getProgramSources() {
	std::vector<std::string> sources = Super::getProgramSources();
	sources.push_back(Common::File::read("EulerHLL.cl"));
	return sources;
}

void EulerHLL::initStep() {
	commands.enqueueNDRangeKernel(calcEigenvaluesKernel, offsetNd, globalSize, localSize);
}

void EulerHLL::calcTimestep() {
	commands.enqueueNDRangeKernel(calcCFLKernel, offsetNd, globalSize, localSize);
	findMinTimestep();	
}

void EulerHLL::step() {
	integrator->integrate([&](cl::Buffer derivBuffer) {
		commands.enqueueNDRangeKernel(calcFluxKernel, offsetNd, globalSize, localSize);
		calcFluxDerivKernel.setArg(0, derivBuffer);
		commands.enqueueNDRangeKernel(calcFluxDerivKernel, offsetNd, globalSize, localSize);
	});

	applyGravity();
}

