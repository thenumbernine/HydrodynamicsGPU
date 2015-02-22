#include "HydroGPU/Solver/EulerHLL.h"
#include "HydroGPU/Equation/Euler.h"
#include "HydroGPU/HydroGPUApp.h"
#include "Common/File.h"

namespace HydroGPU {
namespace Solver {

void EulerHLL::init() {
	Super::init();

	//all Euler and MHD systems also have a separate potential buffer...
	app->setArgs(calcEigenvaluesKernel, eigenvaluesBuffer, stateBuffer, selfgrav->potentialBuffer);
	app->setArgs(calcFluxKernel, fluxBuffer, stateBuffer, eigenvaluesBuffer, selfgrav->potentialBuffer, dtBuffer);
}

void EulerHLL::createEquation() {
	equation = std::make_shared<HydroGPU::Equation::Euler>(this);
}

std::string EulerHLL::getFluxSource() {
	return Common::File::read("EulerHLL.cl");
}

void EulerHLL::step() {
	Super::step();
	selfgrav->applyPotential();
}

}
}

