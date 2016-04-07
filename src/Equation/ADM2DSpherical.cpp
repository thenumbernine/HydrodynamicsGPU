#include "HydroGPU/Equation/ADM2DSpherical.h"
#include "HydroGPU/Boundary/Boundary.h"
#include "HydroGPU/HydroGPUApp.h"
#include "Common/Exception.h"

namespace HydroGPU {
namespace Equation {

enum {
	BOUNDARY_METHOD_NONE = -1,
	BOUNDARY_METHOD_PERIODIC,
	BOUNDARY_METHOD_MIRROR,
	BOUNDARY_METHOD_FREEFLOW,
	NUM_BOUNDARY_METHODS
};

ADM2DSpherical::ADM2DSpherical(HydroGPUApp* app_)
: Super(app_)
{
	//TODO fixme
	displayVariables = std::vector<std::string>{
		"ALPHA",
		"G",
		"A",
		"D",
		"K"
	};

	//matches above
	boundaryMethods = std::vector<std::string>{
		"PERIODIC",
		"MIRROR",
		"FREEFLOW"
	};

	states.push_back("ALPHA");
	states.push_back("G_R_R");
	states.push_back("G_THETA_THETA");
	states.push_back("A");
	states.push_back("B");
	states.push_back("D_A");
	states.push_back("D_B");
	states.push_back("K_A");
	states.push_back("K_B");
	states.push_back("LAMBDA");
}
	
void ADM2DSpherical::getProgramSources(std::vector<std::string>& sources) {
	Super::getProgramSources(sources);

	std::string adm_BonaMasso_f = "1.f";
	app->lua["adm_BonaMasso_f"] >> adm_BonaMasso_f;
	sources[0] += "#define ADM_BONA_MASSO_F " + adm_BonaMasso_f + "\n";
	std::string adm_BonaMasso_df_dalpha = "0.f";
	app->lua["adm_BonaMasso_df_dalpha"] >> adm_BonaMasso_df_dalpha;
	sources[0] += "#define ADM_BONA_MASSO_DF_DALPHA " + adm_BonaMasso_df_dalpha + "\n";
	
	sources.push_back("#include \"ADM2DSphericalCommon.cl\"\n");

	//tell the Roe solver to calculate left & right separately
	// this is slower for dense small matrices (like the Euler equations)
	// but for the ADM, which hold no eigenvector struct data, and compute the eigentransform solely from state data
	// because they are sparse huge matrices, 
	//it saves both speed and memory.
	sources.push_back("#define ROE_EIGENFIELD_TRANSFORM_SEPARATE 1\n");
}

int ADM2DSpherical::stateGetBoundaryKernelForBoundaryMethod(int dim, int state, int minmax) {
	switch (app->boundaryMethods(dim, minmax)) {
	case BOUNDARY_METHOD_NONE:
		return BOUNDARY_KERNEL_NONE;
	case BOUNDARY_METHOD_PERIODIC:
		return BOUNDARY_KERNEL_PERIODIC;
	case BOUNDARY_METHOD_MIRROR:
		return BOUNDARY_KERNEL_MIRROR;	//which states should be negative'd and which shouldn't ...
	case BOUNDARY_METHOD_FREEFLOW:
		return BOUNDARY_KERNEL_FREEFLOW;
	}
	throw Common::Exception() << "got an unknown boundary method " << app->boundaryMethods(dim, minmax) << " for dim " << dim;
}

}
}

