#include "HydroGPU/Solver/EulerHLLC.h"
#include "HydroGPU/HydroGPUApp.h"

namespace HydroGPU {
namespace Solver {

std::string EulerHLLC::getFluxSource() {
	return "#include \"EulerHLLC.cl\"\n";
}

}
}

