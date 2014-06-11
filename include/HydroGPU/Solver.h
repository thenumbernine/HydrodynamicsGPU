#pragma once

#include "HydroGPU/Shared/Common.h"	//cl shared header
#include "Tensor/Vector.h"

struct HydroGPUApp;
struct Solver {
	Solver() {}
	Solver(HydroGPUApp &app) {}
	virtual ~Solver() {}

	virtual void update() = 0;
	virtual void addDrop(Tensor::Vector<float,DIM> pos, Tensor::Vector<float,DIM> vel) = 0;
	virtual void screenshot() = 0;
	virtual void save() = 0;
};

