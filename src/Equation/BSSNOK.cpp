#include "HydroGPU/Equation/BSSNOK.h"
#include "HydroGPU/Solver/Solver.h"
#include "HydroGPU/Boundary/Boundary.h"
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

BSSNOK::BSSNOK(HydroGPU::Solver::Solver* solver_)
: Super(solver_)
{
	displayMethods = std::vector<std::string>{
		"ALPHA",
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
	a_i = partial_i ln alpha									<- n
	Phi_i = partial_i phi										<- n
	dTilde_ijk = 1/2 partial_i gammaTilde_jk					<- n^2 * (n+1)/2
	K = K^i_i													<- 1
	ATilde_ij = exp(-4 phi) (K_ij - 1/3 K gamma_ij)				<- n * (n+1)/2
	GammaTilde^i = gammaTilde^jk connTilde^i_jk					<- n

	num state variables = n^2 * (n+1)/2 + n * (n+1)/2 + 3*n + 1
	for n=3
	num state variables = 18 + 6 + 9 + 1 = 34
	
	... then we remove one for ATilde^i_i = 0	<- ATilde trace-free
	... then we remove three for gammaTilde^jk dTilde_ijk = 0	<- dTilde_i trace-free

	so num state variables = 30
	*/

	int dim = solver->app->dim;

	std::vector<std::string> dimNames = {"X", "Y", "Z"};

	for (int i = 0; i < dim; ++i) {
		states.push_back("A_" + dimNames[i]);
	}
	for (int i = 0; i < dim; ++i) {
		states.push_back("PHI_" + dimNames[i]);
	}
	for (int i = 0; i < dim; ++i) {
		for (int j = 0; j < dim; ++j) {
			for (int k = 0; k <= j; ++k) {
				//and skip the last one of each dTilde_i
				if (j == dim-1 && k == dim-1) break;
				states.push_back("DTILDE_" + dimNames[i] + dimNames[j] + dimNames[k]);
			}
		}
	}
	states.push_back("K");
	for (int i = 0; i < dim; ++i) {
		for (int j = 0; j <= i; ++j) {
			if (i == dim-1 && j == dim-1) break;
			states.push_back("ATILDE_" + dimNames[i] + dimNames[j]);
		}
	}
	for (int i = 0; i < dim; ++i) {
		states.push_back("CONNTILDE_" + dimNames[i]);
	}
}

void BSSNOK::getProgramSources(std::vector<std::string>& sources) {
	Super::getProgramSources(sources);
	
	real adm_BonaMasso_f = 1.f;
	solver->app->lua.ref()["adm_BonaMasso_f"] >> adm_BonaMasso_f;
	sources[0] += "#define BSSNOK_BONA_MASSO_F " + toNumericString<real>(adm_BonaMasso_f) + "\n";
	
	sources.push_back(Common::File::read("BSSNOKCommon.cl"));
}

int BSSNOK::stateGetBoundaryKernelForBoundaryMethod(int dim, int state) {
	switch (solver->app->boundaryMethods(dim)) {
	case BOUNDARY_METHOD_PERIODIC:
		return BOUNDARY_KERNEL_PERIODIC;
		break;
	case BOUNDARY_METHOD_MIRROR:
		return BOUNDARY_KERNEL_MIRROR;
		break;		
	case BOUNDARY_METHOD_FREEFLOW:
		return BOUNDARY_KERNEL_FREEFLOW;
		break;
	}
	throw Common::Exception() << "got an unknown boundary method " << solver->app->boundaryMethods(dim) << " for dim " << dim;
}

}
}


