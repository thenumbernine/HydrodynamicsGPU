#pragma once

#include "HydroGPU/Equation.h"

struct Solver;

struct SRHDEquation : public Equation {
	typedef Equation Super;
	SRHDEquation(Solver& solver);
	virtual void getProgramSources(Solver& solver, std::vector<std::string>& sources);
};

