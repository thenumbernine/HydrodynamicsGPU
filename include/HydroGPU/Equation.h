#pragma once

#include <vector>
#include <string>

struct Solver;

struct Equation {
	std::vector<std::string> displayMethods;
	std::vector<std::string> boundaryMethods;
	std::vector<std::string> states;
	
	Equation();	
	virtual void getProgramSources(Solver& solver, std::vector<std::string>& sources);
	virtual int stateGetBoundaryKernelForBoundaryMethod(Solver& solver, int dim, int state) = 0;
	virtual int gravityGetBoundaryKernelForBoundaryMethod(Solver& solver, int dim) = 0;
	std::string buildEnumCode(const std::string& prefix, const std::vector<std::string>& enumStrs);
};

