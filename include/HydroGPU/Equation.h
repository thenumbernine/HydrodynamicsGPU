#pragma once

#include <vector>
#include <string>

struct Solver;

struct Equation {
	int numStates;
	virtual void getProgramSources(Solver& solver, std::vector<std::string>& sources) = 0;
};

