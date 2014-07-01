#pragma once

#include "HydroGPU/Equation.h"

struct Solver;

struct MHDEquation : public Equation {
	MHDEquation(Solver& solver);
	virtual std::string getSource(Solver& solver);
};

