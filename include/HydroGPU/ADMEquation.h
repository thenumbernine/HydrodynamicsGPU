#pragma once

#include "HydroGPU/Equation.h"

struct Solver;

struct ADMEquation : public Equation {
	ADMEquation(Solver& solver);
	virtual std::string getSource(Solver& solver);
};
