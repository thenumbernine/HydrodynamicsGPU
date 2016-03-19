#include "HydroGPU/Solver/MHDHLLC.h"
#include "HydroGPU/Equation/MHD.h"
#include "HydroGPU/HydroGPUApp.h"

namespace HydroGPU {
namespace Solver {

void MHDHLLC::init() {
	Super::init();
	
	//all Euler and MHD systems also have a separate potential buffer...
	CLCommon::setArgs(calcEigenvaluesKernel, eigenvaluesBuffer, stateBuffer, selfgrav->potentialBuffer);
	CLCommon::setArgs(calcFluxKernel, fluxBuffer, stateBuffer, eigenvaluesBuffer, selfgrav->potentialBuffer);
}

void MHDHLLC::createEquation() {
	equation = std::make_shared<HydroGPU::Equation::MHD>(app);
}

std::string MHDHLLC::getFluxSource() {
	return "#include \"MHDHLLC.cl\"\n";
}

void MHDHLLC::step(real dt) {
	Super::step(dt);
	selfgrav->applyPotential(dt);
	divfree->update();
}

}
}

