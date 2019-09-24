#pragma once

#include "HydroGPU/Solver/SelfGravitation.h"
#include "HydroGPU/Integrator/Integrator.h"
#include "HydroGPU/Shared/Common.h"	//real
#include "HydroGPU/Solver/ISolver.h"
#include "Profiler/Stat.h"
#include "Tensor/Vector.h"
#include "CLCommon/cl.hpp"
#include <vector>
#include <string>
#include <memory>

namespace HydroGPU {
struct HydroGPUApp;
namespace Solver {

struct Solver : public ISolver {
	using Super = ISolver;
	
	friend struct HydroGPU::Integrator::Integrator;
	friend struct HydroGPU::Equation::Equation;

	struct EventProfileEntry {
		EventProfileEntry(std::string name_) : name(name_) {}
		std::string name;
		cl::Event clEvent;
		Profiler::Stat stat;
	};
	
	std::vector<EventProfileEntry*> entries;

	//public for Equation...
	HydroGPUApp *app;

public:	//protected:
	cl::Program program;
	cl::CommandQueue commands;

	/*
	initialized by the child class, but used in arguments in the parent class
	*/
	cl::Buffer stateBuffer;	
	
	/*
	holds the timestep computed based on the wavespeeds at the interface at each grid point 
	*/
	cl::Buffer dtBuffer;
	
	/*
	used for reduction to get a min across the whole dtBuffer.
	only needs to be 1/localsize1d[0] of the size of dtBuffer (courtesy of the first reduction iteration)
	*/
	cl::Buffer dtSwapBuffer;
	
	cl::Kernel findMinTimestepKernel;

	std::vector<std::vector<std::vector<cl::Kernel>>> boundaryKernels;	//[NUM_BOUNDARY_METHODS][app.dim][min/max];

	//construct this after the program has been compiled
	std::shared_ptr<HydroGPU::Integrator::Integrator> integrator;

	//useful to have around
	cl::NDRange globalSize;
	cl::NDRange localSize;
	cl::NDRange localSize1d;
	cl::NDRange offset1d;
	cl::NDRange offsetNd;

	int frame;
	std::shared_ptr<Equation::Equation> equation;

public:
	
	Solver(HydroGPUApp* app);
	virtual ~Solver() {}

	virtual void init();	//...because I'm using virtual function calls in here
	
protected:
	virtual void createEquation() {}
public:
	virtual std::shared_ptr<Equation::Equation> getEquation() const { return equation; }
protected:
	virtual std::vector<std::string> getProgramSources();
	virtual void initBuffers();
	virtual void initKernels();
public:
	int numStates();	//shorthand for equation->states.size()
	virtual int getNumFluxStates();
	int getVolume();	
protected:
	virtual real findMinTimestep();
public:
	virtual void getBoundaryRanges(int dimIndex, cl::NDRange &offset, cl::NDRange &global, cl::NDRange &local);
	virtual void boundary();
protected:

	virtual void initStep();
	virtual real calcTimestep() = 0;
	virtual void step(real dt) = 0;
public:
	virtual void update();

protected:
	//called upon resetState
	//converts stateBuffer and whatever other buffers into CPU-side vectors
	struct Converter {
		Solver* solver;
		std::vector<real> stateVec;
		
		Converter(Solver* solver);

		//how large the stack size for lua is.
		//TODO use some other method to read in info that doesn't need this info.
		//this calls through to the Equation
		//use this instead so it can add extra channels (like potential energy for self-gravitation)
		virtual int numChannels();

		//lua state -> cellResults -> Converter CPU buffer
		//all at once for convenience of compatability with config.lua's initState()
		virtual void setValues(int index, const std::vector<real>& cellValues);
	
		//post readCells: Converter CPU buffer -> Solver GPU buffer
		//call after all setValue calls are done
		virtual void toGPU();
		
		//Solver GPU buffer -> Converter CPU buffer
		//call before any getValue calls
		virtual void fromGPU();

		//Converter CPU buffer -> return individual value
		//one at a time so I can save individual images 
		virtual real getValue(int index, int channel);
	};
	virtual std::shared_ptr<Converter> createConverter();
public:
	virtual void resetState();

protected:
	virtual std::vector<std::string> getSaveChannelNames();
	virtual int getSaveIndex();
public:
	virtual void save();

	//so AMD sucks
	//enqueueFillBuffer is broken on my card.  a simple test case can show this.
	//to work around it I have a separate kernel to do that task.
	//On a separate note, maybe put the offset1d and localSize1d ranges in a separate object, along with this function?
	struct CL {
		CL(Solver* solver_);
		void zero(cl::Buffer buffer, size_t size);
		cl::Buffer alloc(size_t size, const std::string& name = std::string());
	//protected:	
		Solver* solver;
		size_t totalAlloc;
	} cl;
};

}
}
