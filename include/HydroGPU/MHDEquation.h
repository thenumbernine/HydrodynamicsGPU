#pragma once

#include "HydroGPU/Equation.h"

struct Solver;

struct MHDEquation : public Equation {
	typedef Equation Super;
	MHDEquation(Solver& solver);
	virtual void getProgramSources(Solver& solver, std::vector<std::string>& sources);
	virtual int getBoundaryKernelForBoundaryMethod(Solver& solver, int dim, int state);
};

