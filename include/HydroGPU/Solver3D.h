#pragma once

#include "HydroGPU/Solver.h"
#include "Profiler/Stat.h"
#include "Shader/Program.h"
#include "Tensor/Quat.h"
#include "HydroGPU/Shared/Common3D.h"
#include <OpenCL/cl.hpp>
#include <vector>

struct HydroGPUApp;

struct Solver3D : public Solver {
	cl::Program program;
	
	//common kernels for all 3D
	std::vector<std::vector<cl::Kernel>> stateBoundaryKernels;	//[NUM_BOUNDARY_METHODS][DIM];
	cl::Buffer stateBuffer;
	cl::Buffer cflBuffer;
	cl::Buffer cflSwapBuffer;
	cl::Buffer dtBuffer;
	cl::Buffer gravityPotentialBuffer;
	
	cl::Kernel calcCFLMinReduceKernel;
	cl::Kernel convertToTexKernel;
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
	cl::NDRange offsetNd;
	cl::NDRange globalSize;
	cl::NDRange globalWidth;
	cl::NDRange globalHeight;
	cl::NDRange localSize;
	cl::NDRange localSize1d;
	HydroGPUApp &app;
	cl::CommandQueue commands;

	cl::ImageGL fluidTexMem;		//data is written to this buffer before rendering
	GLuint fluidTex;

	std::shared_ptr<Shader::Program> shader;
	Tensor::Quat<float> viewAngle;
	float viewDist;

	Solver3D(HydroGPUApp &app, const std::string &programFilename);
	virtual ~Solver3D();
	
	virtual void resetState(std::vector<real8> stateVec);
	virtual void update();
	virtual void display();
	virtual void resize();

	//input
	virtual void mouseMove(int x, int y, int dx, int dy);
	virtual void mousePan(int dx, int dy);
	virtual void mouseZoom(int dz);

	virtual void addDrop();
	virtual void screenshot();
	virtual void save();

protected:
	virtual void initStep();
	virtual void calcTimestep() = 0;
	virtual void findMinTimestep();
	virtual void step() = 0;
	virtual void boundary();
	virtual void setPoissonRelaxRepeatArg();
};

