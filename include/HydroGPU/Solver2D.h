#pragma once

#include "HydroGPU/Solver.h"
#include "Profiler/Stat.h"
#include <OpenCL/cl.hpp>
#include <vector>

struct HydroGPUApp;

struct Solver2D : public Solver {
	cl::Program program;

	//common kernels for all 2D
	std::vector<std::vector<cl::Kernel>> stateBoundaryKernels;	//[NUM_BOUNDARY_METHODS][DIM];
	cl::Buffer stateBuffer;
	cl::Buffer cflBuffer;
	cl::Buffer cflSwapBuffer;
	cl::Buffer dtBuffer;
	cl::Buffer gravityPotentialBuffer;
	
	cl::Kernel calcCFLMinReduceKernel;
	cl::Kernel convertToTexKernel;
	cl::Kernel addDropKernel;
	cl::Kernel addSourceKernel;
	cl::Kernel poissonRelaxKernel;
	cl::Kernel addGravityKernel;

	struct EventProfileEntry {
		EventProfileEntry(std::string name_) : name(name_) {}
		std::string name;
		cl::Event clEvent;
		Profiler::Stat stat;
	};
	
	std::vector<EventProfileEntry*> entries;

	//useful to have around
	cl::NDRange offset1d;
	cl::NDRange offset2d;
	cl::NDRange globalSize;
	cl::NDRange globalWidth;
	cl::NDRange globalHeight;
	cl::NDRange localSize;
	cl::NDRange localSize1d;
	real2 dx;
	HydroGPUApp &app;
	cl::CommandQueue commands;
	
	//for mouse input
	cl_float2 addSourcePos, addSourceVel;
	
	Solver2D(HydroGPUApp &app, std::vector<real4> stateVec, const std::string &programFilename);
	virtual ~Solver2D();
	
	virtual void addDrop(Tensor::Vector<float,DIM> pos, Tensor::Vector<float,DIM> vel);
	virtual void screenshot();
	virtual void save();	//picks the filename automatically based on what's already there
	virtual void save(std::string filename);

	virtual void update();
protected:
	virtual void boundary();
	virtual void initStep();
	virtual void calcTimestep() = 0;
	virtual void findMinTimestep();
	virtual void step() = 0;

};

