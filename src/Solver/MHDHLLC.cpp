#include "HydroGPU/Solver/MHDHLLC.h"
#include "HydroGPU/Equation/MHD.h"
#include "HydroGPU/HydroGPUApp.h"
#include "Common/File.h"

namespace HydroGPU {
namespace Solver {

void MHDHLLC::init() {
	divfree = std::make_shared<MHDRemoveDivergence>(*this);
	Super::init();
	divfree->init();
}

void MHDHLLC::createEquation() {
	equation = std::make_shared<HydroGPU::Equation::MHD>(*this);
}

std::string MHDHLLC::getFluxSource() {
	return Common::File::read("MHDHLLC.cl");
}

std::vector<std::string> MHDHLLC::getProgramSources() {
	std::vector<std::string> sources = Super::getProgramSources();
	divfree->getProgramSources(sources);
	return sources;
}

void MHDHLLC::step() {
	Super::step();
	divfree->update();
}

}
}

