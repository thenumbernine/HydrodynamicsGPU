#include "HydroGPU/Equation/Equation.h"
#include "HydroGPU/HydroGPUApp.h"

namespace HydroGPU {
namespace Equation {

Equation::Equation(HydroGPUApp* app_) : app(app_) {}

std::string Equation::buildEnumCode(const std::string& prefix, const std::vector<std::string>& enumStrs) {
	std::string str = "enum {\n";
	for (size_t i = 0; i < enumStrs.size(); ++i) {
		str += "\t" + prefix + "_" + enumStrs[i] + ",\n";
	}
	str += "\tNUM_" + prefix + "\n";
	str += "};\n";
	return str;
}

void Equation::getProgramSources(std::vector<std::string>& sources) {
	sources[0] += buildEnumCode("STATE", states);
	sources[0] += buildEnumCode("DISPLAY", displayVariables);
	sources[0] += buildEnumCode("BOUNDARY", boundaryMethods);
	sources[0] += "#define DISPLAY_NONE -1\n";
}

void Equation::readStateCell(real* state, const real* source) {
	for (int i = 0; i < (int)states.size(); ++i) {
		state[i] = source[i];
	}
}

int Equation::numReadStateChannels() {
	return states.size();
}

}
}
