#pragma once

#include "HydroGPU/Solver.h"
#include "Profiler/Stat.h"

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
	cl::Kernel calcFluxKernel;
	cl::Kernel updateStateKernel;
	cl::Kernel convertToTexKernel;
	cl::Kernel addDropKernel;
	
	struct EventProfileEntry {
		EventProfileEntry(std::string name_) : name(name_) {}
		std::string name;
		cl::Event clEvent;
		Profiler::Stat stat;
	};
	
	EventProfileEntry calcEigenDecompositionEvent;
	EventProfileEntry calcCFLAndDeltaQTildeEvent;
	EventProfileEntry calcCFLMinReduceEvent;
	EventProfileEntry calcCFLMinFinalEvent;
	EventProfileEntry calcFluxEvent;
	EventProfileEntry updateStateEvent;
	std::vector<EventProfileEntry*> entries;

	cl::NDRange globalSize, localSize;

	cl_float2 addSourcePos, addSourceVel;
	
	bool useGPU;
	
	real cfl;
	Vector<int,2> size;

	RoeSolver(
		cl::Device device,
		cl::Context context,
		Vector<int,3> size,
		cl::CommandQueue commands,
		real* xmin,
		real* xmax,
		cl_mem fluidTexMem,
		cl_mem gradientTexMem,
		bool useGPU);
	
	virtual ~RoeSolver();

	virtual void update(cl_mem fluidTexMem);
	virtual void addDrop(Vector<float,DIM> pos, Vector<float,DIM> vel);
};


