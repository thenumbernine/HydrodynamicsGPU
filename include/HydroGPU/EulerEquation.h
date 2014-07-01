#pragma once

#include "HydroGPU/Equation.h"

struct Solver;

struct EulerEquation : public Equation {
	EulerEquation(Solver& solver);
	virtual std::string getSource(Solver& solver);
};

