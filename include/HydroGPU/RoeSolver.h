#pragma once

//OpenCL shared header
#include "roe_euler_2d.h"

#include "Profiler/Stat.h"
#include "Tensor/Vector.h"
#include <OpenCL/cl.hpp>
#include <vector>

struct HydroGPUApp;

struct RoeSolver {
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

	HydroGPUApp &app;

	EventProfileEntry calcEigenDecompositionEvent;
	EventProfileEntry calcCFLAndDeltaQTildeEvent;
	EventProfileEntry calcCFLMinReduceEvent;
	EventProfileEntry calcCFLMinFinalEvent;
	EventProfileEntry calcFluxEvent;
	EventProfileEntry updateStateEvent;
	std::vector<EventProfileEntry*> entries;

	cl::NDRange globalSize, localSize;

	cl_float2 addSourcePos, addSourceVel;
	
	real cfl;

	RoeSolver(HydroGPUApp &app);
	virtual ~RoeSolver();

	virtual void update();
	virtual void addDrop(Tensor::Vector<float,DIM> pos, Tensor::Vector<float,DIM> vel);
};


