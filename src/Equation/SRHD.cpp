#include "HydroGPU/Equation/SRHD.h"
#include "HydroGPU/Boundary/Boundary.h"
#include "HydroGPU/toNumericString.h"
#include "HydroGPU/HydroGPUApp.h"
#include "Common/Exception.h"

namespace HydroGPU {
namespace Equation {

SRHD::SRHD(HydroGPUApp* app_)
: Super(app_)
{
	displayVariables.push_back("DENSITY");
	displayVariables.push_back("VELOCITY_X");
	if (app->dim > 1) displayVariables.push_back("VELOCITY_Y");
	if (app->dim > 2) displayVariables.push_back("VELOCITY_Z");
	displayVariables.push_back("VELOCITY_MAGN");
	displayVariables.push_back("E_INTERNAL");
	displayVariables.push_back("P");
	displayVariables.push_back("H");
	displayVariables.push_back("D");
	displayVariables.push_back("S_X");
	if (app->dim > 1) displayVariables.push_back("S_Y");
	if (app->dim > 2) displayVariables.push_back("S_Z");
	displayVariables.push_back("S_MAGN");
	displayVariables.push_back("TAU");

	//matches Equations/SelfGravitationBehavior 
	boundaryMethods = std::vector<std::string>{
		"PERIODIC",
		"MIRROR",
		"FREEFLOW"
	};

	states.push_back("REST_MASS_DENSITY");
	states.push_back("MOMENTUM_DENSITY_X");
	if (app->dim > 1) states.push_back("MOMENTUM_DENSITY_Y");
	if (app->dim > 2) states.push_back("MOMENTUM_DENSITY_Z");
	states.push_back("TOTAL_ENERGY_DENSITY");
}

void SRHD::getProgramSources(std::vector<std::string>& sources) {
	Super::getProgramSources(sources);

	std::vector<std::string> primitives;
	primitives.push_back("DENSITY");
	primitives.push_back("VELOCITY_X");
	if (app->dim > 1) primitives.push_back("VELOCITY_Y");
	if (app->dim > 2) primitives.push_back("VELOCITY_Z");
	primitives.push_back("SPECIFIC_INTERNAL_ENERGY");
	sources[0] += buildEnumCode("PRIMITIVE", primitives);
	
	sources[0] += "#include \"HydroGPU/Shared/Common.h\"\n";	//for real's definition
	
	real gamma = 1.4f;
	app->lua["gamma"] >> gamma;
	sources[0] += "constant real gamma = " + toNumericString<real>(gamma) + ";\n";

//TODO: add iterator to LuaCxx::Ref with key() and value(), and make sure the implicit lua_tostring on the numbers is a good idea ...
	LuaCxx::Ref defs = app->lua["defs"];
	for (LuaCxx::Ref::iterator i = defs.begin(); i != defs.end(); ++i) {	//pairs() kv iterator
		sources[0] += std::string("#define ") + i.key.operator std::string() + std::string(" ") + i.value.operator std::string() + "\n";
	}

	sources.push_back("#include \"SRHDCommon.cl\"\n");
}

int SRHD::stateGetBoundaryKernelForBoundaryMethod(int dim, int state, int minmax) {
	switch (app->boundaryMethods(dim, minmax)) {
	case BOUNDARY_METHOD_NONE:
		return BOUNDARY_KERNEL_NONE;
	case BOUNDARY_METHOD_PERIODIC:
		return BOUNDARY_KERNEL_PERIODIC;
	case BOUNDARY_METHOD_MIRROR:
		return dim + 1 == state ? BOUNDARY_KERNEL_REFLECT : BOUNDARY_KERNEL_MIRROR;
	case BOUNDARY_METHOD_FREEFLOW:
		return BOUNDARY_KERNEL_FREEFLOW;
	}
	throw Common::Exception() << "got an unknown boundary method " << app->boundaryMethods(dim, minmax) << " for dim " << dim;
}

//same as Euler...
void SRHD::readStateCell(real* state, const real* source) {
	state[0] = source[0];
	state[1] = source[1];
	if (app->dim > 1) {
		state[2] = source[2];
	}
	if (app->dim > 2) {
		state[3] = source[3];
	}
	if (states.size() == 8) {
		state[4] = source[4];
		state[5] = source[5];
		state[6] = source[6];
	}
	state[states.size()-1] = source[7];

}

int SRHD::numReadStateChannels() {
	return 8;
}

}
}
