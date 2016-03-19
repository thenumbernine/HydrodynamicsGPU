#include "HydroGPU/Solver/EulerHLL.h"
#include "HydroGPU/Equation/Euler.h"
#include "HydroGPU/HydroGPUApp.h"

namespace HydroGPU {
namespace Solver {

void EulerHLL::init() {
	Super::init();

	//all Euler and MHD systems also have a separate potential buffer...
	CLCommon::setArgs(calcEigenvaluesKernel, eigenvaluesBuffer, stateBuffer, selfgrav->potentialBuffer);
	CLCommon::setArgs(calcFluxKernel, fluxBuffer, stateBuffer, eigenvaluesBuffer, selfgrav->potentialBuffer);
}

void EulerHLL::createEquation() {
	equation = std::make_shared<HydroGPU::Equation::Euler>(app);
}

std::string EulerHLL::getFluxSource() {
	return "#include \"EulerHLL.cl\"\n";
}

void EulerHLL::step(real dt) {
	Super::step(dt);
	selfgrav->applyPotential(dt);
}

}
}

