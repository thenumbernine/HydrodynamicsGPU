#include "HydroGPU/Equation/BSSNOK.h"
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

static std::vector<std::string> sym33suffixes {
	"XX",
	"XY",
	"XZ",
	"YY",
	"YZ",
	"ZZ",
};

BSSNOK::BSSNOK(HydroGPUApp* app_)
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

	displayVariables = std::vector<std::string>{
		"ALPHA",
		"VOLUME",
		"PHI",
		"K",
	};

	//matches above
	boundaryMethods = std::vector<std::string>{
		"PERIODIC",
		"MIRROR",
		"FREEFLOW"
	};
	
	/*
	hyperbolic formalism of BSSNOK:
	a_i = (ln alpha),i = alpha,i / alpha									<- n
	Phi_i = phi,i															<- n
	dTilde_ijk = 1/2 gammaTilde_jk,i										<- n^2 * (n+1)/2
	K = K^i_i																<- 1
	ATilde_ij = exp(-4 phi) (K_ij - 1/3 K gamma_ij)							<- n * (n+1)/2
	GammaTilde^i = -gammaTilde^jk GammaTilde^i_jk = -gammaTilde^ij_,k		<- n

	num state variables = n^2 * (n+1)/2 + n * (n+1)/2 + 3*n + 1
	for n=3
	num state variables = 18 + 6 + 9 + 1 = 34

	partial differential equation (Alcubierre, section 5.6)
	alpha,t = -alpha^2 f K
	phi,t = -1/6 alpha K
	gammaTilde_ij,t =
	a_i,t = -alpha (f K),i
	Phi_i,t = -1/6 alpha K,i
	dTilde_ijk,t = -alpha ATilde_jk,i
	K,t = -alpha exp(-4 phi) gammaTilde^mn a_n,m
	ATilde_ij,t = -alpha exp(-4 phi) LambdaTilde^k_ij,k
	GammaTilde^i,t = -4/3 alpha (gammaTilde^ik K),k
		for LambdaTilde^k_ij = (dTilde^k_ij + delta^k_(i (a_j) - GammaTilde_j) + 2 Phi_j) )^TF
		for TF the trace-free part
	
	constraints:
	A^i_i = 0
	*/

	states.push_back("ALPHA");							//alpha
	addSuffixes(states, "PHI_", spaceSuffixes);			//phi = ln(gamma) / 12
	addSuffixes(states, "GAMMATILDE_", sym33suffixes);	//gammaTilde_ij = exp(-4 phi) gamma_ij = gamma^(1/3) gamma_ij
	addSuffixes(states, "A_", spaceSuffixes);			//a_i = partial_i ln alpha = (partial_i alpha) / alpha
	addSuffixes(states, "PHI_", spaceSuffixes);			//Phi_i = partial_i phi
	addSuffixes(states, "DTILDE_X", sym33suffixes);		//DTilde_ijk = 1/2 partial_i gammaTilde_jk
	addSuffixes(states, "DTILDE_Y", sym33suffixes);
	addSuffixes(states, "DTILDE_Z", sym33suffixes);
	states.push_back("K");								//K = K^i_i
	addSuffixes(states, "ATILDE_", sym33suffixes);		//ATilde_ij = exp(-4 phi) A_ij = exp(-4 phi) (K_ij - 1/3 gamma_ij K)
	addSuffixes(states, "CONNTILDE_", spaceSuffixes);	//GammaTilde^i = gammaTilde^jk GammaTilde^i_jk
}

void BSSNOK::getProgramSources(std::vector<std::string>& sources) {
	Super::getProgramSources(sources);
	
	real adm_BonaMasso_f = 1.f;
	app->lua["adm_BonaMasso_f"] >> adm_BonaMasso_f;
	sources[0] += "#define BSSNOK_BONA_MASSO_F " + toNumericString<real>(adm_BonaMasso_f) + "\n";
	
	sources.push_back("#include \"BSSNOKCommon.cl\"\n");
}

int BSSNOK::stateGetBoundaryKernelForBoundaryMethod(int dim, int state, int minmax) {
	switch (app->boundaryMethods(dim, minmax)) {
	case BOUNDARY_METHOD_NONE:
		return BOUNDARY_KERNEL_NONE;
	case BOUNDARY_METHOD_PERIODIC:
		return BOUNDARY_KERNEL_PERIODIC;
	case BOUNDARY_METHOD_MIRROR:
		return BOUNDARY_KERNEL_MIRROR;
	case BOUNDARY_METHOD_FREEFLOW:
		return BOUNDARY_KERNEL_FREEFLOW;
	}
	throw Common::Exception() << "got an unknown boundary method " << app->boundaryMethods(dim, minmax) << " for dim " << dim;
}

}
}
