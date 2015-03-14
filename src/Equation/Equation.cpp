#include "HydroGPU/Equation/Equation.h"
#include "HydroGPU/Solver/Solver.h"

namespace HydroGPU {
namespace Equation {

Equation::Equation(HydroGPU::Solver::Solver* solver_) : solver(solver_) {}

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
	sources[0] += buildEnumCode("DISPLAY", displayMethods);
	sources[0] += buildEnumCode("BOUNDARY", boundaryMethods);
}

void Equation::readStateCell(real* state, const real* source) {
	for (int i = 0; i < (int)states.size(); ++i) {
		state[i] = source[i];
	}
}

int Equation::numReadStateChannels() {
	return solver->numStates();
}

}
}

