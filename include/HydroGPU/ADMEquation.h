#pragma once

#include "HydroGPU/Equation.h"
#include <vector>
#include <string>

struct Solver;

struct ADMEquation : public Equation {
	ADMEquation(Solver& solver);
	virtual void getProgramSources(Solver& solver, std::vector<std::string>& sources);
};
