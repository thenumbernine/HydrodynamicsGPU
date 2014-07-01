#pragma once

#include <string>

struct Solver;

struct Equation {
	int numStates;
	virtual std::string getSource(Solver& solver) = 0;
};

