#include "HydroGPU/Solver/Roe.h"
#include "HydroGPU/HydroGPUApp.h"

namespace HydroGPU {
namespace Solver {

Roe::Roe(
	HydroGPUApp* app_)
: Super(app_)
, calcEigenBasisSideEvent("calcEigenBasisSide")
, findMinTimestepEvent("findMinTimestep")
, calcDeltaQTildeEvent("calcDeltaQTilde")
, calcFluxEvent("calcFlux")
{
}

void Roe::init() {
	Super::init();
	
	cl::Context context = app->context;

	entries.push_back(&calcEigenBasisSideEvent);
	if (!app->useFixedDT) {
		entries.push_back(&findMinTimestepEvent);
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
	
	calcEigenBasisSideKernel = cl::Kernel(program, "calcEigenBasisSide");
	app->setArgs(calcEigenBasisSideKernel, eigenvaluesBuffer, eigenfieldsBuffer, stateBuffer);

	findMinTimestepKernel = cl::Kernel(program, "findMinTimestep");
	app->setArgs(findMinTimestepKernel, dtBuffer, eigenvaluesBuffer);
	
	calcDeltaQTildeKernel = cl::Kernel(program, "calcDeltaQTilde");
	app->setArgs(calcDeltaQTildeKernel, deltaQTildeBuffer, eigenfieldsBuffer, stateBuffer);
	
	calcFluxKernel = cl::Kernel(program, "calcFlux");
	app->setArgs(calcFluxKernel, fluxBuffer, stateBuffer, eigenvaluesBuffer, eigenfieldsBuffer, deltaQTildeBuffer);

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
	calcEigenBasisSideKernel.setArg(5, side);
	commands.enqueueNDRangeKernel(calcEigenBasisSideKernel, offsetNd, globalSize, localSize, nullptr, &calcEigenBasisSideEvent.clEvent);
}

void Roe::initStep() {
	for (int side = 0; side < app->dim; ++side) {
		initFluxSide(side);
	}
}

real Roe::calcTimestep() {
	//TODO for each side: 
	//	calc side's interface primitives
	//	calc eigenvalues for side based on primitives
	//	calc dt for eigenvalues
	//use min of all side's dt's
	
	commands.enqueueNDRangeKernel(findMinTimestepKernel, offsetNd, globalSize, localSize, nullptr, &findMinTimestepEvent.clEvent);
	return findMinTimestep();
}

void Roe::step(real dt) {
	calcFluxKernel.setArg(5, dt);
	for (int sideIndex = 0; sideIndex < 2 * app->dim - 1; ++sideIndex) {
		
		int side = sideIndex;
		if (side >= app->dim) side = 2 * app->dim - 2 - side;
	
		//TODO except first iteration:
		//	calc interface primitives
		//	calc eigenvalues (not needed for first sideIndex) and eigenvectors
		//(unless the interface primtive buffer only stores one side, then they'll need to be recalculated,
		//	...unless the calcTimestep() function cycles through all sides from last to first)
		
		if (sideIndex > 0) {
			initFluxSide(side);
		}
			
		integrator->integrate(
			side == app->dim-1 ? dt : (.5f * dt),
			[&](cl::Buffer derivBuffer) {
				calcDeriv(dt, derivBuffer, side);
			}
		);
	}
}

void Roe::calcDeriv(real dt, cl::Buffer derivBuffer, int side) {
	calcDeltaQTildeKernel.setArg(3, side);
	commands.enqueueNDRangeKernel(calcDeltaQTildeKernel, offsetNd, globalSize, localSize, nullptr, &calcDeltaQTildeEvent.clEvent);
	
	calcFlux(dt, side);
	
	calcFluxDerivKernel.setArg(0, derivBuffer);
	calcFluxDerivKernel.setArg(2, side);
	commands.enqueueNDRangeKernel(calcFluxDerivKernel, offsetNd, globalSize, localSize);
}

void Roe::calcFlux(real dt, int side) {
	calcFluxKernel.setArg(5, dt);
	calcFluxKernel.setArg(6, side);
	commands.enqueueNDRangeKernel(calcFluxKernel, offsetNd, globalSize, localSize, nullptr, &calcFluxEvent.clEvent);
}

}
}

