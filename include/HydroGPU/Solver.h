#pragma once

#include "TensorMath/Vector.h"
#include <OpenCL/cl.hpp>
#include <vector>

//OpenCL shared header
#include "roe_euler_2d.h"

struct Solver {
	Solver() {}
	Solver(
		cl::Device device,
		cl::Context context,
		Vector<int,3> size,
		cl::CommandQueue commands,
		std::vector<Cell> &cells,
		real *xmin,
		real *xmax,
		cl_mem fluidTexMem,
		cl_mem gradientTexMem,
		bool useGPU) {}

	virtual ~Solver() {}

	virtual void update(cl_mem fluidTexMem) = 0;
	virtual void addDrop(Vector<float,DIM> pos, Vector<float,DIM> vel) = 0;
};

