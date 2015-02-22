#include "HydroGPU/Solver/MHDHLLC.h"
#include "HydroGPU/Equation/MHD.h"
#include "HydroGPU/HydroGPUApp.h"
#include "Common/File.h"

namespace HydroGPU {
namespace Solver {

void MHDHLLC::init() {
	Super::init();
	
	//all Euler and MHD systems also have a separate potential buffer...
	app->setArgs(calcEigenvaluesKernel, eigenvaluesBuffer, stateBuffer, selfgrav->potentialBuffer);
	app->setArgs(calcFluxKernel, fluxBuffer, stateBuffer, eigenvaluesBuffer, selfgrav->potentialBuffer, dtBuffer);
}

void MHDHLLC::createEquation() {
	equation = std::make_shared<HydroGPU::Equation::MHD>(this);
}

std::string MHDHLLC::getFluxSource() {
	return Common::File::read("MHDHLLC.cl");
}

void MHDHLLC::step() {
	Super::step();
	selfgrav->applyPotential();
	divfree->update();
}

}
}

