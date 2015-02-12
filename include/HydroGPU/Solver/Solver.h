#pragma once

#include "HydroGPU/Equation/Equation.h"
#include "HydroGPU/Integrator/Integrator.h"
#include "HydroGPU/Shared/Common.h"	//cl shared header
#include "Profiler/Stat.h"
#include "Tensor/Vector.h"
#include <OpenCL/cl.hpp>
#include <vector>
#include <string>
#include <memory>

namespace HydroGPU {
struct HydroGPUApp;
namespace Plot {
struct Plot;
struct VectorField;
}
namespace Solver {

struct Solver {
	friend struct HydroGPU::Integrator::Integrator;

	struct EventProfileEntry {
		EventProfileEntry(std::string name_) : name(name_) {}
		std::string name;
		cl::Event clEvent;
		Profiler::Stat stat;
	};
	
	std::vector<EventProfileEntry*> entries;
	cl::ImageGL fluidTexMem;		//data is written to this buffer before rendering

	//public for Equation...
	HydroGPUApp &app;
	
public:	//protected:
	cl::Program program;
	cl::CommandQueue commands;

	cl::Buffer stateBuffer;	//initialized by the child class, but used in arguments in the parent class
	cl::Buffer cflBuffer;
	cl::Buffer cflSwapBuffer;
	cl::Buffer dtBuffer;
	cl::Buffer potentialBuffer;
	
	cl::Kernel calcCFLMinReduceKernel;
	cl::Kernel poissonRelaxKernel;
	cl::Kernel calcGravityDerivKernel;
	cl::Kernel convertToTexKernel;

	std::vector<std::vector<cl::Kernel>> boundaryKernels;	//[NUM_BOUNDARY_METHODS][app.dim];

	//construct this after the program has been compiled
	std::shared_ptr<HydroGPU::Integrator::Integrator> integrator;
	std::shared_ptr<HydroGPU::Plot::VectorField> vectorField;
	std::shared_ptr<HydroGPU::Plot::Plot> plot;

	//useful to have around
	cl::NDRange globalSize;
	cl::NDRange localSize;
	cl::NDRange localSize1d;
	cl::NDRange offset1d;
	cl::NDRange offsetNd;

	size_t totalAlloc;
public:
	std::shared_ptr<HydroGPU::Equation::Equation> equation;

	Solver(HydroGPUApp& app);
	virtual ~Solver() {}

	virtual void init();	//...because I'm using virtual function calls in here
protected:
	virtual void createEquation() = 0;
public:	//protected:
	virtual std::vector<std::string> getProgramSources();
	virtual void initKernels();
	
	int numStates();	//shorthand
	int getVolume();
	
	cl::Buffer clAlloc(size_t size);

	virtual void applyPotential();

	virtual void findMinTimestep();

	virtual void getBoundaryRanges(int dimIndex, cl::NDRange &offset, cl::NDRange &global, cl::NDRange &local);
	virtual void boundary();
	virtual void potentialBoundary();

	virtual void calcTimestep() = 0;
	virtual void initStep();
	virtual void step() = 0;
public:
	virtual void update();
	
	virtual void display();
	virtual void resize();

	virtual void mouseMove(int x, int y, int dx, int dy);
	virtual void mousePan(int dx, int dy);
	virtual void mouseZoom(int dz);

	virtual void resetState();
	virtual void addDrop();
	virtual void screenshot();
	virtual void save();
};

}
}

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

