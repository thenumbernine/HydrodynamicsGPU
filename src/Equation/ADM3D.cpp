#include "HydroGPU/Equation/ADM3D.h"
#include "HydroGPU/Solver/Solver.h"
#include "HydroGPU/HydroGPUApp.h"
#include "Common/File.h"
#include "Common/Exception.h"

namespace HydroGPU {
namespace Equation {

enum {
	BOUNDARY_METHOD_PERIODIC,
	BOUNDARY_METHOD_MIRROR,
	BOUNDARY_METHOD_FREEFLOW,
	NUM_BOUNDARY_METHODS
};

static std::vector<std::string> spaceSuffixes {"X", "Y", "Z"};

static std::vector<std::string> sym33suffixes {
	"XX",
	"YY",
	"ZZ",
	"XY",
	"XZ",
	"YZ",
};

static void addStatesWithSuffix(
	std::vector<std::string>& states,
	const std::string& variable,
	const std::vector<std::string>& suffixes)
{
	std::for_each(suffixes.begin(), suffixes.end(), [&](const std::string& field) {
		states.push_back(variable + field);
	});
}

ADM3D::ADM3D(HydroGPU::Solver::Solver* solver_)
: Super(solver_)
{
	displayMethods = std::vector<std::string>{
		"LAPSE",
		"VOLUME",
		"EXTRINSIC_CURVATURE"
	};

	//matches above
	boundaryMethods = std::vector<std::string>{
		"PERIODIC",
		"MIRROR",
		"FREEFLOW"
	};
	
	//you can factor these out someday ...
	states.push_back("ALPHA");
	addStatesWithSuffix(states, "GAMMA_", sym33suffixes);
	//these form the hyperbolic system ...
	addStatesWithSuffix(states, "A_", spaceSuffixes);	//A_i = partial_i alpha
	addStatesWithSuffix(states, "D_X", sym33suffixes);	//D_kij = 1/2 partial_k gamma_ij
	addStatesWithSuffix(states, "D_Y", sym33suffixes);
	addStatesWithSuffix(states, "D_Z", sym33suffixes);
	addStatesWithSuffix(states, "K_", sym33suffixes);	//extrinsic curvature
	addStatesWithSuffix(states, "V_", spaceSuffixes);	//V_k = D_km^m - D^m_mk
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
		std::for_each(sym33suffixes.begin(), sym33suffixes.end(), [&](const std::string& suffix){
			sources[0] += "#define SYM33_" + suffix + " " + std::to_string(i) + "\n";
			++i;
		});
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

	sources.push_back(Common::File::read("ADM3DCommon.cl"));
}

int ADM3D::stateGetBoundaryKernelForBoundaryMethod(int dim, int state) {
	switch (solver->app->boundaryMethods(dim)) {
	case BOUNDARY_METHOD_PERIODIC:
		return BOUNDARY_KERNEL_PERIODIC;
		break;
	case BOUNDARY_METHOD_MIRROR:
		return BOUNDARY_KERNEL_MIRROR;	//which states should be negative'd and which shouldn't ...
		break;		
	case BOUNDARY_METHOD_FREEFLOW:
		return BOUNDARY_KERNEL_FREEFLOW;
		break;
	}
	throw Common::Exception() << "got an unknown boundary method " << solver->app->boundaryMethods(dim) << " for dim " << dim;
}

int ADM3D::gravityGetBoundaryKernelForBoundaryMethod(int dim) {
	switch (solver->app->boundaryMethods(dim)) {
	case BOUNDARY_METHOD_PERIODIC:
		return BOUNDARY_KERNEL_PERIODIC;
		break;
	case BOUNDARY_METHOD_MIRROR:
		return BOUNDARY_KERNEL_FREEFLOW;
		break;		
	case BOUNDARY_METHOD_FREEFLOW:
		return BOUNDARY_KERNEL_FREEFLOW;
		break;
	}
	throw Common::Exception() << "got an unknown boundary method " << solver->app->boundaryMethods(dim) << " for dim " << dim;
}

}
}


