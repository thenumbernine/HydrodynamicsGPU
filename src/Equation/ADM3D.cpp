#include "HydroGPU/Equation/ADM3D.h"
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
	//derived values
	displayVariables.push_back("K");
	displayVariables.push_back("GAMMA");
	displayVariables.push_back("VOLUME");
	displayVariables.push_back("EXPANSION");
	displayVariables.push_back("GRAVITY_MAGN");
//	displayVariables.push_back("GAUSSIAN_CURVATURE");
	//constraints
	addSuffixes(displayVariables, "A_ALPHA_CONSTRAINT_", spaceSuffixes);	//A_i = partial_i ln alpha
	addSuffixes(displayVariables, "D_X_GAMMA_CONSTRAINT_", sym33Suffixes);	//D_kij = 1/2 partial_k gamma_ij
	addSuffixes(displayVariables, "D_Y_GAMMA_CONSTRAINT_", sym33Suffixes);	//D_kij = 1/2 partial_k gamma_ij
	addSuffixes(displayVariables, "D_Z_GAMMA_CONSTRAINT_", sym33Suffixes);	//D_kij = 1/2 partial_k gamma_ij
	addSuffixes(displayVariables, "V_CONSTRAINT_", spaceSuffixes);	//V^i = D^im_m - D_m^mi
	
	//TODO more constraints:
	//displayVariables.push_back("CONSTRAINT_HAMILTONIAN");
	//addSuffixes(displayVariables, "CONSTRAINT_MOMENTUM_", spaceSuffixes);
	//addSuffixes(displayVariables, "CONSTRAINT_EFE_", sym33Suffixes);


	addSuffixes(vectorFieldVars, "GAMMA_", spaceSuffixes);	//visualize 3 axis of the gamma metric ... separately.  TODO all 3 at once would be nice
	vectorFieldVars.push_back("A");
	//TODO how to visualize D's ... do you show the tetrad of the metric's deriv for each vector sepraately? or do you show the grad of deriv for each component separately? all?
	// D_[i]jk, or D_i[j]k ? nine of each means 18 total ...
	addSuffixes(vectorFieldVars, "K_", spaceSuffixes);	//likewise with extrinsic curvature
	vectorFieldVars.push_back("V");
	vectorFieldVars.push_back("GRAVITY");
	//vectorFieldVars.push_back("TIDAL");
	vectorFieldVars.push_back("V_CONSTRAINT");
	//vectorFieldVars.push_back("MOMENTUM_CONSTRAINT");
}

void ADM3D::getProgramSources(std::vector<std::string>& sources) {
	Super::getProgramSources(sources);
	
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
