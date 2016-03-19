#include "HydroGPU/Equation/ADM3D.h"
#include "HydroGPU/Boundary/Boundary.h"
#include "HydroGPU/toNumericString.h"
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

static std::vector<std::string> sym33Suffixes {
	"XX",
	"XY",
	"XZ",
	"YY",
	"YZ",
	"ZZ",
};

ADM3D::ADM3D(HydroGPUApp* app_)
: Super(app_)
{
	std::function<void(std::vector<std::string>&, const std::string&, const std::vector<std::string>&)> addSuffixes = [&](
		std::vector<std::string>& strs,
		const std::string& prefix,
		const std::vector<std::string>& suffixes)
	{
		for (const std::string& field : suffixes) {
			strs.push_back(prefix + field);
		}
	};

	//matches above
	boundaryMethods = std::vector<std::string>{
		"PERIODIC",
		"MIRROR",
		"FREEFLOW"
	};

	//you can factor these out someday ...
	states.push_back("ALPHA");
	addSuffixes(states, "GAMMA_", sym33Suffixes);
	//these form the hyperbolic system ...
	addSuffixes(states, "A_", spaceSuffixes);	//A_i = partial_i ln alpha
	addSuffixes(states, "D_X", sym33Suffixes);	//D_kij = 1/2 partial_k gamma_ij
	addSuffixes(states, "D_Y", sym33Suffixes);
	addSuffixes(states, "D_Z", sym33Suffixes);
	addSuffixes(states, "K_", sym33Suffixes);	//extrinsic curvature
	addSuffixes(states, "V_", spaceSuffixes);	//V_k = D_km^m - D^m_mk. TODO replace with Gamma^k = Gamma^k_ij gamma^ij as the book describes

	states.push_back("DENSITY");
	addSuffixes(states, "VELOCITY_", spaceSuffixes);
	states.push_back("PRESSURE");

	/*
	it'd be nice to have trees on the display variables:
	states
		alpha
		gamma_ij
			... x6
		k_ij
			... x6
		a_i
			... x3
		d_ijk
			... x18 (maybe break these down too?)
		v_i
			... x3
	aux values
		k
		gamma
		volume
	constraints
		a_i vs alpha
			... x3
		d_ijk vs gamma_ij
			... x18
		v_i vs d_ijk gamma^jk
			... x3
	
	...but how to specify nodes?
	*/
	displayVariables = states;
	displayVariables.push_back("K");
	displayVariables.push_back("GAMMA");
	displayVariables.push_back("VOLUME");
	addSuffixes(displayVariables, "A_ALPHA_CONSTRAINT_", spaceSuffixes);	//A_i = partial_i ln alpha
	addSuffixes(displayVariables, "D_X_GAMMA_CONSTRAINT_", sym33Suffixes);	//D_kij = 1/2 partial_k gamma_ij
	addSuffixes(displayVariables, "D_Y_GAMMA_CONSTRAINT_", sym33Suffixes);	//D_kij = 1/2 partial_k gamma_ij
	addSuffixes(displayVariables, "D_Z_GAMMA_CONSTRAINT_", sym33Suffixes);	//D_kij = 1/2 partial_k gamma_ij
	addSuffixes(displayVariables, "V_CONSTRAINT_", spaceSuffixes);	//V^i = D^im_m - D_m^mi
	
	//TODO more constraints:
	//displayVariables.push_back("CONSTRAINT_HAMILTONIAN");
	//addSuffixes(displayVariables, "CONSTRAINT_MOMENTUM_", spaceSuffixes);
	//addSuffixes(displayVariables, "CONSTRAINT_EFE_", sym33Suffixes);
}

void ADM3D::getProgramSources(std::vector<std::string>& sources) {
	Super::getProgramSources(sources);
	
	//TODO detect type, cast number to CL string or use literal string
	//if type is number ...
	//real adm_BonaMasso_f = 1.f;
	//app->lua.ref()["adm_BonaMasso_f"] >> adm_BonaMasso_f;
	//sources[0] += "#define ADM_BONA_MASSO_F " + toNumericString<real>(adm_BonaMasso_f) + "\n";
	//else if type is string ...
	std::string adm_BonaMasso_f = "1.f";
	app->lua.ref()["adm_BonaMasso_f"] >> adm_BonaMasso_f;
	sources[0] += "#define ADM_BONA_MASSO_F " + adm_BonaMasso_f + "\n";
	
	std::string adm_BonaMasso_df_dalpha = "0.f";
	app->lua.ref()["adm_BonaMasso_df_dalpha"] >> adm_BonaMasso_df_dalpha;
	sources[0] += "#define ADM_BONA_MASSO_DF_DALPHA " + adm_BonaMasso_df_dalpha + "\n";

	//speed of light is used for coordinate transform from 3-vel to 4-vel with the stress-energy tensor
	//for the R4_ij and G_ij components
	real speedOfLight = 299792458;	//default in m/s
	app->lua.ref()["speedOfLight"] >> speedOfLight;
	sources[0] += "#define SPEED_OF_LIGHT " + toNumericString<real>(speedOfLight) + "\n";
	
	{
		int i = 0;
		for (const std::string& suffix : sym33Suffixes){
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
	// but for the ADM, which hold no eigenvector struct data, and compute the eigentransform solely from state data
	// because they are sparse huge matrices, 
	//it saves both speed and memory.
	sources.push_back("#define ROE_EIGENFIELD_TRANSFORM_SEPARATE 1\n");
}

int ADM3D::stateGetBoundaryKernelForBoundaryMethod(int dim, int state, int minmax) {
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
