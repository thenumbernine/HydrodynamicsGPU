#pragma once

#include <OpenCL/OpenCL.h>
#include <vector>

//OpenCL shared header
#include "roe_euler_2d.h"

struct Solver {
	Solver(
		cl_device_id deviceID, 
		cl_context context, 
		cl_int2 size, 
		cl_command_queue commands,
		std::vector<Cell> &cells,
		real2 xmin,
		real2 xmax,
		cl_mem fluidTexMem,
		cl_mem gradientTexMem,
		size_t *local_size,
		bool useGPU) {}

	virtual ~Solver() {}

	virtual void update(
		cl_command_queue commands, 
		cl_mem fluidTexMem, 
		size_t *global_size,
		size_t *local_size) = 0;
};

