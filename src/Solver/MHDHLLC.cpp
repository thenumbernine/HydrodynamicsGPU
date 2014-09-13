#include "HydroGPU/Solver/MHDHLLC.h"
#include "HydroGPU/Equation/MHD.h"
#include "HydroGPU/HydroGPUApp.h"
#include "Common/File.h"

namespace HydroGPU {
namespace Solver {

MHDHLLC::MHDHLLC(
	HydroGPUApp& app_)
: Super(app_)
{
	equation = std::make_shared<HydroGPU::Equation::MHD>(*this);
}
std::string MHDHLLC::getFluxSource() {
	return Common::File::read("MHDHLLC.cl");
}

}
}

