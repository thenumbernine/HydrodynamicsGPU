#pragma once

#include "HydroGPU/Equation.h"
#include <vector>
#include <string>

struct Solver;

struct ADMEquation : public Equation {
	typedef Equation Super;
	ADMEquation(Solver& solver);
	virtual void getProgramSources(Solver& solver, std::vector<std::string>& sources);
	virtual int stateGetBoundaryKernelForBoundaryMethod(Solver& solver, int dim, int state);
	virtual int gravityGetBoundaryKernelForBoundaryMethod(Solver& solver, int dim);
};
