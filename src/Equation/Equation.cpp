#include "HydroGPU/Equation/Equation.h"

namespace HydroGPU {
namespace Equation {

Equation::Equation() {}

std::string Equation::buildEnumCode(const std::string& prefix, const std::vector<std::string>& enumStrs) {
	std::string str = "enum {\n";
	for (size_t i = 0; i < enumStrs.size(); ++i) {
		str += "\t" + prefix + "_" + enumStrs[i] + ",\n";
	}
	str += "\tNUM_" + prefix + "\n";
	str += "};\n";
	return str;
}

void Equation::getProgramSources(Solver& solver, std::vector<std::string>& sources) {
	sources[0] += buildEnumCode("STATE", states);
	sources[0] += buildEnumCode("DISPLAY", displayMethods);
	sources[0] += buildEnumCode("BOUNDARY", boundaryMethods);
}

}
}

