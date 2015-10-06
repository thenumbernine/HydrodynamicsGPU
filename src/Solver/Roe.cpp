#include "HydroGPU/Solver/Roe.h"
#include "HydroGPU/HydroGPUApp.h"

namespace HydroGPU {
namespace Solver {

Roe::Roe(
	HydroGPUApp* app_)
: Super(app_)
{
}

void Roe::initBuffers() {
	Super::initBuffers();

	int volume = getVolume();

	eigenvaluesBuffer = clAlloc(sizeof(real) * getEigenSpaceDim() * volume, "Roe::eigenvaluesBuffer");
	eigenfieldsBuffer = clAlloc(sizeof(real) * getEigenTransformStructSize() * volume, "Roe::eigenfieldsBuffer");
	deltaQTildeBuffer = clAlloc(sizeof(real) * getEigenSpaceDim() * volume, "Roe::deltaQTildeBuffer");
	fluxBuffer = clAlloc(sizeof(real) * numStates() * volume, "Roe::fluxBuffer");

	//zero flux
	commands.enqueueFillBuffer(fluxBuffer, 0.f, 0, sizeof(real) * numStates() * volume);
}

int Roe::getEigenTransformStructSize() {
	return getEigenSpaceDim() * getEigenSpaceDim() * 2;	//times two for forward and inverse
}

int Roe::getEigenSpaceDim() {
	return numStates();
}

void Roe::initKernels() {
	Super::initKernels();

	calcEigenBasisSideKernel = cl::Kernel(program, "calcEigenBasisSide");
	CLCommon::setArgs(calcEigenBasisSideKernel, eigenvaluesBuffer, eigenfieldsBuffer, stateBuffer);
	
	findMinTimestepKernel = cl::Kernel(program, "findMinTimestep");
	CLCommon::setArgs(findMinTimestepKernel,
			dtBuffer,
//Hydrodynamics ii
#if 1
			eigenvaluesBuffer);
#endif
//Toro 16.38
#if 0
			stateBuffer);
#endif	

	calcDeltaQTildeKernel = cl::Kernel(program, "calcDeltaQTilde");
	CLCommon::setArgs(calcDeltaQTildeKernel, deltaQTildeBuffer, eigenfieldsBuffer, stateBuffer);
	
	calcFluxKernel = cl::Kernel(program, "calcFlux");
	CLCommon::setArgs(calcFluxKernel, fluxBuffer, stateBuffer, eigenvaluesBuffer, eigenfieldsBuffer, deltaQTildeBuffer); 
	
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
	
void Roe::initFluxSide(int side) {
	calcEigenBasisSideKernel.setArg(3, side);
	commands.enqueueNDRangeKernel(calcEigenBasisSideKernel, offsetNd, globalSize, localSize);
}

real Roe::calcTimestep() {
	real dt = std::numeric_limits<real>::infinity();
	for (int side = 0; side < app->dim; ++side) {
		initFluxSide(side);
		
		findMinTimestepKernel.setArg(2, side);
		commands.enqueueNDRangeKernel(findMinTimestepKernel, offsetNd, globalSize, localSize);
		
		dt = std::min(dt, findMinTimestep());
	}
	return dt;
}

void Roe::step(real dt) {
	int sideStart, sideEnd, sideStep;
	getSideRange(sideStart, sideEnd, sideStep);
	for (int side = sideStart; side != sideEnd; side += sideStep) {
		initFluxSide(side);
		integrator->integrate(dt, [&](cl::Buffer derivBuffer) {
			calcDeriv(derivBuffer, dt, side);
		});
	}
}

void Roe::calcDeriv(cl::Buffer derivBuffer, real dt, int side) {
	calcDeltaQTildeKernel.setArg(3, side);
	commands.enqueueNDRangeKernel(calcDeltaQTildeKernel, offsetNd, globalSize, localSize);
	
	calcFlux(dt, side);
	
	calcFluxDerivKernel.setArg(0, derivBuffer);
	calcFluxDerivKernel.setArg(2, side);
	commands.enqueueNDRangeKernel(calcFluxDerivKernel, offsetNd, globalSize, localSize);
}

void Roe::calcFlux(real dt, int side) {
	calcFluxKernel.setArg(5, dt);
	calcFluxKernel.setArg(6, side);
	commands.enqueueNDRangeKernel(calcFluxKernel, offsetNd, globalSize, localSize);
}

}
}

