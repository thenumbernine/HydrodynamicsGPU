#pragma once

#include "HydroGPU/Equation.h"

struct Solver;

struct EulerEquation : public Equation {
	EulerEquation(Solver& solver);
	virtual void getProgramSources(Solver& solver, std::vector<std::string>& sources);
};

