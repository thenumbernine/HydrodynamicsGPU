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
	cl::Kernel applyBoundaryHorizontalKernel;
	cl::Kernel applyBoundaryVerticalKernel;
	cl::Kernel calcInterfaceVelocityKernel;
	cl::Kernel calcFluxKernel;
	cl::Kernel integrateFluxKernel;
	cl::Kernel computePressureKernel;
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
	EventProfileEntry applyBoundaryHorizontalEvent;
	EventProfileEntry applyBoundaryVerticalEvent;
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

	BurgersSolver(HydroGPUApp &app);
	virtual ~BurgersSolver();

	virtual void update();
	virtual void addDrop(Tensor::Vector<float,DIM> pos, Tensor::Vector<float,DIM> vel);
	virtual void screenshot();
	virtual void save();
};


