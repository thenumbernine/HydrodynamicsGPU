#pragma once

#include "HydroGPU/Equation.h"
#include "HydroGPU/Shared/Common.h"	//cl shared header
#include "Tensor/Vector.h"
#include <OpenCL/cl.hpp>
#include <vector>
#include <string>
#include <memory>

struct HydroGPUApp;
struct Solver {
	//public for Equation...
	HydroGPUApp &app;
	
protected:
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
	
	std::vector<std::vector<cl::Kernel>> boundaryKernels;	//[NUM_BOUNDARY_METHODS][app.dim];

	//useful to have around
	cl::NDRange globalSize;
	cl::NDRange localSize;
	cl::NDRange localSize1d;
	cl::NDRange offset1d;
	cl::NDRange offsetNd;

	size_t totalAlloc;
public:
	std::shared_ptr<Equation> equation;

	Solver(HydroGPUApp& app);
	virtual ~Solver() {}

	virtual void init();	//...because I'm using virtual function calls in here
protected:
	virtual std::vector<std::string> getProgramSources();
	virtual void initKernels();
	cl::Buffer clAlloc(size_t size);
	
	virtual void findMinTimestep();

	virtual void getBoundaryRanges(int dimIndex, cl::NDRange &offset, cl::NDRange &global, cl::NDRange &local);
	virtual void boundary();
	virtual void gravityPotentialBoundary();

	virtual void initStep();
	virtual void step() = 0;
	virtual void calcTimestep() = 0;
public:
	virtual void update();
	
	virtual void display() = 0;
	virtual void resize() = 0;

	virtual void mouseMove(int x, int y, int dx, int dy) = 0;
	virtual void mousePan(int dx, int dy) = 0;
	virtual void mouseZoom(int dz) = 0;

	virtual void resetState();
	virtual void addDrop() = 0;
	virtual void screenshot() = 0;
	virtual void save() = 0;
};


//used by enough folks:

template<typename T> std::string toNumericString(T value);

template<> inline std::string toNumericString<double>(double value) {
	std::string s = std::to_string(value);
	if (s.find("e") == std::string::npos) {
		if (s.find(".") == std::string::npos) {
			s += ".";
		}
	}
	return s;
}

template<> inline std::string toNumericString<float>(float value) {
	return toNumericString<double>(value) + "f";
}

enum {
	BOUNDARY_KERNEL_PERIODIC,
	BOUNDARY_KERNEL_MIRROR,		//	\_ combined to make up reflecting boundary conditions
	BOUNDARY_KERNEL_REFLECT,	//	/
	BOUNDARY_KERNEL_FREEFLOW,
	NUM_BOUNDARY_KERNELS
};

