#pragma once

#include "HydroGPU/Equation.h"

struct Solver;

struct MHDEquation : public Equation {
	MHDEquation(Solver& solver);
	virtual void getProgramSources(Solver& solver, std::vector<std::string>& sources);
};

