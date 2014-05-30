#pragma once

#include "HydroGPU/Solver.h"

struct RoeSolver : public Solver {
	cl::Program program;
	cl::CommandQueue commands;
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
	cl::Kernel addDropKernel;

	cl::NDRange globalSize, localSize;

	cl_float2 addSourcePos, addSourceVel;
	
	bool useGPU;
	
	real cfl;
	cl_int2 size;

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
		bool useGPU);

	virtual void update(cl_mem fluidTexMem);
	virtual void addDrop(Vector<float,DIM> pos, Vector<float,DIM> vel);
};


