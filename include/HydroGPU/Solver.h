#pragma once

#include "Tensor/Vector.h"
#include <OpenCL/cl.hpp>
#include <vector>

//OpenCL shared header
#include "roe_euler_2d.h"

struct Solver {
	Solver() {}
	Solver(
		cl::Device device,
		cl::Context context,
		Tensor::Vector<int,3> size,
		cl::CommandQueue commands,
		real *xmin,
		real *xmax,
		cl_mem fluidTexMem,
		cl_mem gradientTexMem,
		bool useGPU) {}

	virtual ~Solver() {}

	virtual void update(cl_mem fluidTexMem) = 0;
	virtual void addDrop(Tensor::Vector<float,DIM> pos, Tensor::Vector<float,DIM> vel) = 0;
};

