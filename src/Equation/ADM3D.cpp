#include "HydroGPU/Equation/ADM3D.h"
#include "HydroGPU/Solver/Solver.h"
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

static std::vector<std::string> spaceSuffixes {"X", "Y", "Z"};

static std::vector<std::string> sym33suffixes {
	"XX",
	"XY",
	"XZ",
	"YY",
	"YZ",
	"ZZ",
};

ADM3D::ADM3D(HydroGPU::Solver::Solver* solver_)
: Super(solver_)
{
	displayVariables = std::vector<std::string>{
		"ALPHA",
		"VOLUME",
		"K"
	};

	//matches above
	boundaryMethods = std::vector<std::string>{
		"PERIODIC",
		"MIRROR",
		"FREEFLOW"
	};

	std::function<void(const std::string&, const std::vector<std::string>&)> addStatesWithSuffix = [&](
		const std::string& variable,
		const std::vector<std::string>& suffixes)
	{
		for (const std::string& field : suffixes) {
			states.push_back(variable + field);
		}
	};

	//you can factor these out someday ...
	states.push_back("ALPHA");
	addStatesWithSuffix("GAMMA_", sym33suffixes);
	//these form the hyperbolic system ...
	addStatesWithSuffix("A_", spaceSuffixes);	//A_i = partial_i alpha
	addStatesWithSuffix("D_X", sym33suffixes);	//D_kij = 1/2 partial_k gamma_ij
	addStatesWithSuffix("D_Y", sym33suffixes);
	addStatesWithSuffix("D_Z", sym33suffixes);
	addStatesWithSuffix("K_", sym33suffixes);	//extrinsic curvature
	addStatesWithSuffix("V_", spaceSuffixes);	//V_k = D_km^m - D^m_mk
}

void ADM3D::getProgramSources(std::vector<std::string>& sources) {
	Super::getProgramSources(sources);

	//TODO detect type, cast number to CL string or use literal string
	//if type is number ...
	//real adm_BonaMasso_f = 1.f;
	//solver->app->lua.ref()["adm_BonaMasso_f"] >> adm_BonaMasso_f;
	//sources[0] += "#define ADM_BONA_MASSO_F " + toNumericString<real>(adm_BonaMasso_f) + "\n";
	//else if type is string ...
	std::string adm_BonaMasso_f = "1.f";
	solver->app->lua.ref()["adm_BonaMasso_f"] >> adm_BonaMasso_f;
	sources[0] += "#define ADM_BONA_MASSO_F " + adm_BonaMasso_f + "\n";
	std::string adm_BonaMasso_df_dalpha = "0.f";
	solver->app->lua.ref()["adm_BonaMasso_df_dalpha"] >> adm_BonaMasso_df_dalpha;
	sources[0] += "#define ADM_BONA_MASSO_DF_DALPHA " + adm_BonaMasso_df_dalpha + "\n";

	{
		int i = 0;
		for (const std::string& suffix : sym33suffixes){
			sources[0] += "#define SYM33_" + suffix + " " + std::to_string(i) + "\n";
			++i;
		}
	}

	//and shorthand for the suffix states
	sources[0] += "#define STATE_GAMMA STATE_GAMMA_XX\n";
	sources[0] += "#define STATE_A STATE_A_X\n";
	sources[0] += "#define STATE_D STATE_D_XXX\n";
	sources[0] += "#define STATE_D_X STATE_D_XXX\n";
	sources[0] += "#define STATE_D_Y STATE_D_YXX\n";
	sources[0] += "#define STATE_D_Z STATE_D_ZXX\n";
	sources[0] += "#define STATE_K STATE_K_XX\n";
	sources[0] += "#define STATE_V STATE_V_X\n";

	sources.push_back("#include \"ADM3DCommon.cl\"\n");
	
	//tell the Roe solver to calculate left & right separately
	// this is slower for dense small matrices (like the Euler equations)
	// but for the ADM, which hold no eigenfield struct data, and compute the eigentransform solely from state data
	// because they are sparse huge matrices, 
	//it saves both speed and memory.
	sources.push_back("#define ROE_EIGENFIELD_TRANSFORM_SEPARATE 1\n");
}

int ADM3D::stateGetBoundaryKernelForBoundaryMethod(int dim, int state, int minmax) {
	switch (solver->app->boundaryMethods(dim, minmax)) {
	case BOUNDARY_METHOD_NONE:
		return BOUNDARY_KERNEL_NONE;
	case BOUNDARY_METHOD_PERIODIC:
		return BOUNDARY_KERNEL_PERIODIC;
	case BOUNDARY_METHOD_MIRROR:
		return BOUNDARY_KERNEL_MIRROR;	//which states should be negative'd and which shouldn't ...
	case BOUNDARY_METHOD_FREEFLOW:
		return BOUNDARY_KERNEL_FREEFLOW;
	}
	throw Common::Exception() << "got an unknown boundary method " << solver->app->boundaryMethods(dim, minmax) << " for dim " << dim;
}

}
}


