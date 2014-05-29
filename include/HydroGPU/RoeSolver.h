#pragma once

#include "HydroGPU/Solver.h"

struct RoeSolver : public Solver {
	cl_program program;
	cl_mem cellsMem;		//our main OpenCL buffer for the simulation
	cl_mem cflMem;
	cl_mem cflTimestepMem;
	cl_kernel calcEigenDecompositionKernel;
	cl_kernel calcCFLAndDeltaQTildeKernel;
	cl_kernel calcCFLMinReduceKernel;
	cl_kernel calcCFLMinFinalKernel;
	cl_kernel calcRTildeKernel;
	cl_kernel calcFluxKernel;
	cl_kernel updateStateKernel;
	cl_kernel convertToTexKernel;
	cl_int2 size;
	bool useGPU;
	
	real cfl;

	RoeSolver(
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
		bool useGPU);

	virtual ~RoeSolver();

	virtual void update(
		cl_command_queue commands, 
		cl_mem fluidTexMem, 
		size_t *global_size,
		size_t *local_size); 
};


