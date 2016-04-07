#include "HydroGPU/Solver/FiniteVolumeSolver.h"
#include "HydroGPU/HydroGPUApp.h"

namespace HydroGPU {
namespace Solver {

FiniteVolumeSolver::FiniteVolumeSolver(HydroGPUApp* app_)
: Super(app_) {}

void FiniteVolumeSolver::initBuffers() {
	Super::initBuffers();
	
	fluxBuffer = cl.alloc(sizeof(real) * getNumFluxStates() * getVolume() * app->dim, "FiniteVolumeSolver::fluxBuffer");
	cl.zero(fluxBuffer, getNumFluxStates() * getVolume() * app->dim * sizeof(real));
}

void FiniteVolumeSolver::initKernels() {
	Super::initKernels();

	calcFluxKernel = cl::Kernel(program, "calcFlux");
	CLCommon::setArgs(calcFluxKernel, fluxBuffer, stateBuffer);
	
	calcFluxDerivKernel = cl::Kernel(program, "calcFluxDeriv");
	calcFluxDerivKernel.setArg(1, fluxBuffer);
}

std::vector<std::string> FiniteVolumeSolver::getCalcFluxDerivProgramSources() {
	return {
		"#include \"CalcFluxDeriv.cl\"\n"
	};
}

std::vector<std::string> FiniteVolumeSolver::getProgramSources() {
	std::vector<std::string> sources = Super::getProgramSources();
	std::vector<std::string> added = getCalcFluxDerivProgramSources();
	sources.insert(sources.end(), added.begin(), added.end());
	return sources;
}

}
}
