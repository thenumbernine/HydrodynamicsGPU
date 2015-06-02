#include "HydroGPU/Solver/Roe.h"
#include "HydroGPU/HydroGPUApp.h"

namespace HydroGPU {
namespace Solver {

Roe::Roe(
	HydroGPUApp* app_)
: Super(app_)
, calcEigenBasisEvent("calcEigenBasis")
, calcCFLEvent("calcCFL")
, calcDeltaQTildeEvent("calcDeltaQTilde")
, calcFluxEvent("calcFlux")
{
}

void Roe::init() {
	Super::init();
	
	cl::Context context = app->context;

	entries.push_back(&calcEigenBasisEvent);
	if (!app->useFixedDT) {
		entries.push_back(&calcCFLEvent);
	}
	entries.push_back(&calcDeltaQTildeEvent);
	entries.push_back(&calcFluxEvent);
}

void Roe::initBuffers() {
	Super::initBuffers();

	int volume = getVolume();

	eigenvaluesBuffer = clAlloc(sizeof(real) * getEigenSpaceDim() * volume * app->dim, "Roe::eigenvaluesBuffer");
	eigenfieldsBuffer = clAlloc(sizeof(real) * getEigenTransformStructSize() * volume * app->dim, "Roe::eigenfieldsBuffer");
	deltaQTildeBuffer = clAlloc(sizeof(real) * getEigenSpaceDim() * volume * app->dim, "Roe::deltaQTildeBuffer");
	fluxBuffer = clAlloc(sizeof(real) * numStates() * volume * app->dim, "Roe::fluxBuffer");

	//zero flux
	commands.enqueueFillBuffer(fluxBuffer, 0.f, 0, sizeof(real) * numStates() * volume * app->dim);
}

int Roe::getEigenTransformStructSize() {
	return getEigenSpaceDim() * getEigenSpaceDim() * 2;	//times two for forward and inverse
}

int Roe::getEigenSpaceDim() {
	return numStates();
}

void Roe::initKernels() {
	Super::initKernels();
	
	calcEigenBasisKernel = cl::Kernel(program, "calcEigenBasis");
	app->setArgs(calcEigenBasisKernel, eigenvaluesBuffer, eigenfieldsBuffer, stateBuffer);

	calcCFLKernel = cl::Kernel(program, "calcCFL");
	app->setArgs(calcCFLKernel, cflBuffer, eigenvaluesBuffer, app->cfl);
	
	calcDeltaQTildeKernel = cl::Kernel(program, "calcDeltaQTilde");
	app->setArgs(calcDeltaQTildeKernel, deltaQTildeBuffer, eigenfieldsBuffer, stateBuffer);
	
	calcFluxKernel = cl::Kernel(program, "calcFlux");
	app->setArgs(calcFluxKernel, fluxBuffer, stateBuffer, eigenvaluesBuffer, eigenfieldsBuffer, deltaQTildeBuffer, dtBuffer);
	
	calcFluxDerivKernel = cl::Kernel(program, "calcFluxDeriv");
	calcFluxDerivKernel.setArg(1, fluxBuffer);
}	

std::vector<std::string> Roe::getProgramSources() {
	std::vector<std::string> sources = Super::getProgramSources();
	sources.push_back("#define EIGEN_TRANSFORM_STRUCT_SIZE "+std::to_string(getEigenTransformStructSize())+"\n");
	sources.push_back("#define EIGEN_SPACE_DIM "+std::to_string(getEigenSpaceDim())+"\n");
	std::vector<std::string> added = getEigenProgramSources();
	sources.insert(sources.end(), added.begin(), added.end());
	sources.push_back("#include \"Roe.cl\"\n");
	return sources;
}

std::vector<std::string> Roe::getEigenProgramSources() {
	return {
		"#include \"RoeEigenfieldLinear.cl\"\n"
	};
}

void Roe::initStep() {
	commands.enqueueNDRangeKernel(calcEigenBasisKernel, offsetNd, globalSize, localSize, nullptr, &calcEigenBasisEvent.clEvent);
}

void Roe::calcTimestep() {
	commands.enqueueNDRangeKernel(calcCFLKernel, offsetNd, globalSize, localSize, nullptr, &calcCFLEvent.clEvent);
	findMinTimestep();
}

void Roe::step() {
	for (int side = 0; side < app->dim; ++side) {
		//every time we integrate along one axis, it can potentially change the eigenbasis of the next axis
		//the "initStep" call already calced the eigenbasis once for the timestep, so we don't need to do this for side==0
		// however for every subsequent side, we should update the eigenbasis
		//NOTICE: if our dt is calculated based on side==0 then it might not necessarily be accurate for subsequent sides ...
		if (side > 0) {
			initStep();	//calcEigenBasisKernel and ... for MHDRoe ... clear flux flags
		}

		integrator->integrate([&](cl::Buffer derivBuffer) {
			calcDeriv(derivBuffer, side);
		});
	}
}

void Roe::calcDeriv(cl::Buffer derivBuffer, int side) {
	calcDeltaQTildeKernel.setArg(3, side);
	commands.enqueueNDRangeKernel(calcDeltaQTildeKernel, offsetNd, globalSize, localSize, nullptr, &calcDeltaQTildeEvent.clEvent);
	
	calcFlux(side);
	
	calcFluxDerivKernel.setArg(0, derivBuffer);
	calcFluxDerivKernel.setArg(2, side);
	commands.enqueueNDRangeKernel(calcFluxDerivKernel, offsetNd, globalSize, localSize);
}

void Roe::calcFlux(int side) {
	calcFluxKernel.setArg(6, side);
	commands.enqueueNDRangeKernel(calcFluxKernel, offsetNd, globalSize, localSize, nullptr, &calcFluxEvent.clEvent);
}

}
}

