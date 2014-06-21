#pragma once

#include "HydroGPU/Shared/Common.h"	//cl shared header
#include "Tensor/Vector.h"
#include <OpenCL/cl.hpp>
#include <vector>

struct HydroGPUApp;
struct Solver {
	HydroGPUApp &app;
	
	cl::Program program;
	cl::CommandQueue commands;

	cl::Buffer stateBuffer;	//initialized by the child class, but used in arguments in the parent class
	cl::Buffer cflBuffer;
	cl::Buffer cflSwapBuffer;
	cl::Buffer dtBuffer;
	cl::Buffer gravityPotentialBuffer;
	
	cl::Kernel calcCFLMinReduceKernel;
	cl::Kernel poissonRelaxKernel;
	cl::Kernel addGravityKernel;
	
	std::vector<std::vector<cl::Kernel>> stateBoundaryKernels;	//[NUM_BOUNDARY_METHODS][app.dim];

	//useful to have around
	cl::NDRange globalSize;
	cl::NDRange localSize;
	cl::NDRange localSize1d;
	cl::NDRange offset1d;
	cl::NDRange offsetNd;

	Solver(HydroGPUApp& app, const std::vector<std::string>& programFilenames);
	virtual ~Solver() {}
	
	virtual void initKernels();
	
	virtual void findMinTimestep();
	virtual void setPoissonRelaxRepeatArg();

	virtual void initStep();
	virtual void boundary() = 0;
	virtual void step() = 0;
	virtual void calcTimestep() = 0;
	virtual void update();
	
	virtual void display() = 0;
	virtual void resize() = 0;

	virtual void mouseMove(int x, int y, int dx, int dy) = 0;
	virtual void mousePan(int dx, int dy) = 0;
	virtual void mouseZoom(int dz) = 0;

	virtual void resetState(std::vector<real8> stateVec) = 0;
	virtual void addDrop() = 0;
	virtual void screenshot() = 0;
	virtual void save() = 0;
};

