#include "HydroGPU/Equation/MHD.h"
#include "HydroGPU/Solver/SelfGravitationBehavior.h"
#include "HydroGPU/Solver/MHDRemoveDivergenceBehavior.h"
#include "HydroGPU/toNumericString.h"
#include "HydroGPU/HydroGPUApp.h"
#include "Common/Exception.h"
#include <cassert>

//shared between MHD and EMHD.  put it somewhere everyone can get to it.
template<typename T>
static std::vector<T> append(const std::vector<T>& a, const std::vector<T>& b) {
	std::vector<T> c = a;
	c.insert(c.end(), b.begin(), b.end());
	return c;
}

namespace HydroGPU {
namespace Equation {

MHD::MHD(HydroGPUApp* app_)
: Super(app_, 3)
{
	displayVariables = append(displayVariables, std::vector<std::string>{
		"ENERGY_MAGNETIC",
		"MAGNETIC_FIELD",
		"MAGNETIC_FIELD_X",
		"MAGNETIC_FIELD_Y",
		"MAGNETIC_FIELD_Z",
		"MAGNETIC_DIVERGENCE_BUFFER",
		"MAGNETIC_DIVERGENCE_CALCULATED",
		"MAGNETIC_DIVERGENCE_ERROR",
	});


	states = append(states, std::vector<std::string>{
		"MAGNETIC_FIELD_X",
		"MAGNETIC_FIELD_Y",
		"MAGNETIC_FIELD_Z",
	});
	//move energy_total to the end. idk why.
	assert(states[4] == "ENERGY_TOTAL");
	states.erase(states.begin() + 4);
	states.push_back("ENERGY_TOTAL");
	assert(states.size() == 8);
}

void MHD::getProgramSources(std::vector<std::string>& sources) {
	
	sources[0] += "#define MHD 1\n";	//for EulerMHDCommon.cl
	//precompute the sqrt
	sources[0] += std::string("#define mhd_sqrt_vacuumPermeability ") + toNumericString<real>((real)sqrt((real)app->lua["defs"]["mhd_vacuumPermeability"])) + std::string("\n");
	sources.push_back("#include \"MHDCommon.cl\"\n");
	
	Super::getProgramSources(sources);
}

int MHD::stateGetBoundaryKernelForBoundaryMethod(int dim, int stateIndex, int minmax) {
	if (app->boundaryMethods(dim, minmax) == BOUNDARY_METHOD_MIRROR) {
		return (dim + 1 == stateIndex || dim + 4 == stateIndex) ? BOUNDARY_KERNEL_REFLECT : BOUNDARY_KERNEL_MIRROR;
	}
	return Super::stateGetBoundaryKernelForBoundaryMethod(dim, stateIndex, minmax);
}

void MHD::setupConvertToTexKernelArgs(cl::Kernel convertToTexKernel, Solver::Solver* solver) {
	Super::setupConvertToTexKernelArgs(convertToTexKernel, solver);

	Solver::MHDRemoveDivergenceInterface* mhdSolver = dynamic_cast<Solver::MHDRemoveDivergenceInterface*>(solver);
	assert(mhdSolver != nullptr);
	convertToTexKernel.setArg(5, mhdSolver->getMagneticFieldDivergenceBuffer());
}

}
}
