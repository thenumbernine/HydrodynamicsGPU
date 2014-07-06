#include "HydroGPU/Equation.h"

Equation::Equation()
: numStates(0)
{}

std::string Equation::buildEnumCode(const std::string& prefix, const std::vector<std::string>& enumStrs) {
	std::string str = "enum {\n";
	for (size_t i = 0; i < enumStrs.size(); ++i) {
		std::string comma = i == enumStrs.size()-1 ? "" : ",";
		str += "\t" + prefix + "_" + enumStrs[i] + comma + "\n";
	}
	str += "};\n";
	return str;
}

