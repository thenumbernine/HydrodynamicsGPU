#pragma once

#include "HydroGPU/Shared/Common.h"	//cl shared header
#include "Tensor/Vector.h"
#include <vector>

struct HydroGPUApp;
struct Solver {
	Solver() {}
	Solver(HydroGPUApp &app) {}
	virtual ~Solver() {}

	virtual void update() = 0;
	virtual void display() = 0;
	virtual void resize() = 0;

	virtual void mouseMove(int x, int y, int dx, int dy) = 0;
	virtual void mousePan(int dx, int dy) = 0;
	virtual void mouseZoom(int dz) = 0;

	virtual void resetState(std::vector<real8> stateVec) = 0;
	virtual void addDrop() = 0;
	virtual void screenshot() = 0;
	virtual void save() = 0;
};

