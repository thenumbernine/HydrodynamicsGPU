#include "HydroGPU/Solver/EulerHLLC.h"
#include "HydroGPU/HydroGPUApp.h"
#include "Common/File.h"

namespace HydroGPU {
namespace Solver {

std::string EulerHLLC::getFluxSource() {
	return Common::File::read("EulerHLLC.cl");
}

}
}

