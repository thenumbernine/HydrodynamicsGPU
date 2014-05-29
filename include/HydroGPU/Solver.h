#pragma once

#include <OpenCL/cl.hpp>
#include <vector>

//OpenCL shared header
#include "roe_euler_2d.h"

struct Solver {
	Solver(
		cl::Device device,
		cl::Context context,
		cl_int2 size,
		cl::CommandQueue commands,
		std::vector<Cell> &cells,
		real *xmin,
		real *xmax,
		cl_mem fluidTexMem,
		cl_mem gradientTexMem,
		size_t *local_size,
		bool useGPU) {}

	virtual ~Solver() {}

	virtual void update(
		cl::CommandQueue commands,
		cl_mem fluidTexMem, 
		size_t *global_size,
		size_t *local_size) = 0;
	
	virtual void addDrop(float x, float y, float dx, float dy) = 0;
};

