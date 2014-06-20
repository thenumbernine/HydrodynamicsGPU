#pragma once

#include "HydroGPU/Solver.h"
#include "Profiler/Stat.h"
#include "Shader/Program.h"
#include <OpenCL/cl.hpp>
#include <vector>

struct HydroGPUApp;

struct Solver2D : public Solver {
	typedef Solver Super;
	
	cl::Kernel convertToTexKernel;
	cl::Kernel addDropKernel;
	cl::Kernel addSourceKernel;

	struct EventProfileEntry {
		EventProfileEntry(std::string name_) : name(name_) {}
		std::string name;
		cl::Event clEvent;
		Profiler::Stat stat;
	};
	
	std::vector<EventProfileEntry*> entries;

	cl::ImageGL fluidTexMem;		//data is written to this buffer before rendering
	GLuint fluidTex;
	
	std::shared_ptr<Shader::Program> shader1d;
	
	//for mouse input
	cl_float2 addSourcePos, addSourceVel;
	
	Solver2D(HydroGPUApp &app, const std::string &programFilename);
	virtual ~Solver2D();
	
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
	virtual void boundary();

};

