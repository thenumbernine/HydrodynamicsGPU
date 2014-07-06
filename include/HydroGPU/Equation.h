#pragma once

#include <vector>
#include <string>

struct Solver;

struct Equation {
	int numStates;
	std::vector<std::string> displayMethods;
	std::vector<std::string> boundaryMethods;
	
	Equation();	
	virtual void getProgramSources(Solver& solver, std::vector<std::string>& sources) = 0;
	virtual int getBoundaryKernelForBoundaryMethod(Solver& solver, int dim, int state) = 0;
	std::string buildEnumCode(const std::string& prefix, const std::vector<std::string>& enumStrs);
};

