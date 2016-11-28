#include "HydroGPU/Equation/EMHD.h"

namespace HydroGPU {
namespace Equation {

//vector::operator+, anyone?
template<typename T>
static std::vector<T> append(const std::vector<T>& a, const std::vector<T>& b) {
	std::vector<T> c = a;
	c.insert(c.end(), b.begin(), b.end());
	return c;
}

EMHD::EMHD(HydroGPUApp* app_)
: Super(app_), euler(app_, 3), maxwell(app_)
{
	//TODO euler doesn't add dimensions that aren't present ... but EMHD needs all dimensions
	displayVariables = append(euler.displayVariables, maxwell.displayVariables);
	boundaryMethods = euler.boundaryMethods;	//should match maxwell's boundaryMethods
	states = append(euler.states, maxwell.states);		//how is this used? 
	vectorFieldVars = append(euler.vectorFieldVars, maxwell.vectorFieldVars);
}

int EMHD::stateGetBoundaryKernelForBoundaryMethod(int dim, int state, int minmax) {
	if (state < (int)euler.states.size()) {
		return euler.stateGetBoundaryKernelForBoundaryMethod(dim, state, minmax);
	}
	state -= euler.states.size();
	
	if (state < (int)maxwell.states.size()) {
		return maxwell.stateGetBoundaryKernelForBoundaryMethod(dim, state, minmax);
	}
	state -= maxwell.states.size();

	throw Common::Exception() << "here";
}

void EMHD::readStateCell(real* state, const real* source) {
}

}
}
