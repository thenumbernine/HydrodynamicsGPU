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
	cl::NDRange localSize;
	cl::NDRange localSize1d;
	HydroGPUApp &app;
	cl::CommandQueue commands;

	cl::ImageGL fluidTexMem;		//data is written to this buffer before rendering
	GLuint fluidTex;
	
	//for mouse input
	cl_float2 addSourcePos, addSourceVel;
	
	Solver2D(HydroGPUApp &app, const std::string &programFilename);
	virtual ~Solver2D();
	
	virtual void update();
	virtual void display();
	virtual void resize();

	//input
	virtual void mouseMove(int x, int y, int dx, int dy);
	virtual void mousePan(int dx, int dy);
	virtual void mouseZoom(int dz);

	float viewZoom;
	Tensor::Vector<float,2> viewPos;
	Tensor::Vector<real,2> mousePos, mouseVel;
	
	virtual void addDrop();
	virtual void screenshot();
	virtual void save();	//picks the filename automatically based on what's already there
	virtual void save(std::string filename);

protected:
	virtual void resetState(std::vector<real8> stateVec);
	virtual void initStep();
	virtual void calcTimestep() = 0;
	virtual void findMinTimestep();
	virtual void step() = 0;
	virtual void boundary();
	virtual void setPoissonRelaxRepeatArg();

};

