#pragma once

#include "HydroGPU/Solver.h"
#include "Profiler/Stat.h"
#include "Tensor/Vector.h"
#include <OpenCL/cl.hpp>
#include <vector>

struct HydroGPUApp;

struct BurgersSolver : public Solver {
	cl::Program program;
	cl::CommandQueue commands;
	
	cl::Buffer stateBuffer;
	cl::Buffer interfaceVelocityBuffer;
	cl::Buffer fluxBuffer;
	cl::Buffer pressureBuffer;
	cl::Buffer cflBuffer;
	cl::Buffer cflSwapBuffer;
	cl::Buffer dtBuffer;
	cl::Buffer gravityPotentialBuffer;

	cl::Kernel calcCFLKernel;
	cl::Kernel calcCFLMinReduceKernel;
	std::vector<std::vector<cl::Kernel>> stateBoundaryKernels;//[NUM_BOUNDARY_METHODS][DIM];
	
	cl::Kernel calcInterfaceVelocityKernel;
	cl::Kernel calcInterfaceVelocityHorizontalKernel;
	cl::Kernel calcInterfaceVelocityVerticalKernel;
	
	cl::Kernel calcFluxKernel;
	cl::Kernel calcFluxHorizontalKernel;
	cl::Kernel calcFluxVerticalKernel;
	
	cl::Kernel integrateFluxKernel;
	cl::Kernel computePressureKernel;
	cl::Kernel computePressureHorizontalKernel;
	cl::Kernel computePressureVerticalKernel;
	cl::Kernel diffuseMomentumKernel;
	cl::Kernel diffuseWorkKernel;
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

	HydroGPUApp &app;

	EventProfileEntry calcCFLEvent;
	EventProfileEntry calcCFLMinReduceEvent;
	EventProfileEntry calcInterfaceVelocityEvent;
	EventProfileEntry calcFluxEvent;
	EventProfileEntry integrateFluxEvent;
	EventProfileEntry computePressureEvent;
	EventProfileEntry diffuseMomentumEvent;
	EventProfileEntry diffuseWorkEvent;
	EventProfileEntry addSourceEvent;
	EventProfileEntry poissonRelaxEvent;
	EventProfileEntry addGravityEvent;
	
	std::vector<EventProfileEntry*> entries;

	cl::NDRange globalSize, localSize;

	//for mouse input
	cl_float2 addSourcePos, addSourceVel;

	BurgersSolver(HydroGPUApp &app, std::vector<real4> stateVec);
	virtual ~BurgersSolver();

	virtual void update();
	virtual void addDrop(Tensor::Vector<float,DIM> pos, Tensor::Vector<float,DIM> vel);
	virtual void screenshot();
	virtual void save();

	virtual void apply1DBoundary(cl::Buffer buffer);
};


