#pragma once

#include "HydroGPU/Solver.h"

struct RoeSolver : public Solver {
	cl::Program program;
	cl::Buffer cellsMem;		//our main OpenCL buffer for the simulation
	cl::Buffer cflMem;
	cl::Buffer cflTimestepMem;
	cl::Kernel calcEigenDecompositionKernel;
	cl::Kernel calcCFLAndDeltaQTildeKernel;
	cl::Kernel calcCFLMinReduceKernel;
	cl::Kernel calcCFLMinFinalKernel;
	cl::Kernel calcRTildeKernel;
	cl::Kernel calcFluxKernel;
	cl::Kernel updateStateKernel;
	cl::Kernel convertToTexKernel;
	cl_int2 size;
	bool useGPU;
	
	real cfl;

	RoeSolver(
		cl::Device device,
		cl::Context context,
		cl_int2 size,
		cl::CommandQueue commands,
		std::vector<Cell> &cells,
		real* xmin,
		real* xmax,
		cl_mem fluidTexMem,
		cl_mem gradientTexMem,
		size_t *local_size,
		bool useGPU);

	virtual void update(
		cl::CommandQueue commands, 
		cl_mem fluidTexMem, 
		size_t *global_size,
		size_t *local_size);

	virtual void addDrop(float x, float y, float dx, float dy);
};


