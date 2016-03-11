#include "HydroGPU/Equation/Euler.h"
#include "HydroGPU/Solver/EulerRoe.h"
#include "HydroGPU/HydroGPUApp.h"

namespace HydroGPU {
namespace Solver {

void EulerRoe::initKernels() {
	Super::initKernels();
	
	//all Euler and MHD systems also have a separate potential buffer...
	calcEigenBasisKernel.setArg(3, selfgrav->potentialBuffer);
	calcEigenBasisKernel.setArg(4, selfgrav->solidBuffer);
	calcCellTimestepKernel.setArg(2, selfgrav->solidBuffer);
	calcDeltaQTildeKernel.setArg(3, selfgrav->solidBuffer);
	calcFluxKernel.setArg(6, selfgrav->solidBuffer);
	calcFluxDerivKernel.setArg(2, selfgrav->solidBuffer);
}

void EulerRoe::createEquation() {
	equation = std::make_shared<HydroGPU::Equation::Euler>(this);
}

std::vector<std::string> EulerRoe::getProgramSources() {
	std::vector<std::string> sources = Super::getProgramSources();
	sources.insert(sources.begin(), "#define SOLID 1\n");
	sources.push_back("#include \"EulerRoe.cl\"\n");
	return sources;
}
	
void EulerRoe::step(real dt) {
	Super::step(dt);
	selfgrav->applyPotential(dt);
}

}
}
