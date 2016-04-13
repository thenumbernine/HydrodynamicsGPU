#include "HydroGPU/Solver/EulerHLL.h"
#include "HydroGPU/Equation/Euler.h"
#include "HydroGPU/HydroGPUApp.h"

namespace HydroGPU {
namespace Solver {

void EulerHLL::initKernels() {
	Super::initKernels();

	//all Euler and MHD systems also have a separate potential buffer...
	calcEigenvaluesKernel.setArg(2, selfgrav->potentialBuffer);
	
	calcFluxKernel.setArg(3, selfgrav->potentialBuffer);
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
