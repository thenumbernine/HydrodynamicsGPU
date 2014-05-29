#pragma once

#include "HydroGPU/RoeSolver.h"
#include "GLApp/GLApp.h" 
#include "TensorMath/Vector.h"
#include <OpenCL/cl.hpp>

struct HydroGPUApp : public GLApp {
	bool useGPU;	//whether we want to request the GPU or CPU.  required prior to init()
	
	GLuint fluidTex;
	GLuint gradientTex;
	
	cl::Device device;
	cl::Context context;
	
	cl::CommandQueue commands;
	cl_mem fluidTexMem;		//data is written to this buffer before rendering
	cl_mem gradientTexMem;	//as it is written, data is read from this for mapping values to colors

	Solver *solver;

	Vector<size_t,DIM> local_size;
	Vector<size_t,DIM> global_size;
	cl_int2 size;
	  
	bool leftButtonDown;
	bool leftShiftDown;
	bool rightShiftDown;
	Vector<real,2> xmin, xmax;
	int doUpdate;	//0 = no, 1 = continuous, 2 = single step
	Vector<int,2> screenSize;
	float viewZoom;
	Vector<float,2> viewPos;

	HydroGPUApp();

	cl::Platform getPlatform();
	cl::Device getDevice(cl::Platform platform);

	virtual int main(std::vector<std::string> args);
	virtual void init();
	virtual void shutdown();
	virtual void resize(int width, int height);
	virtual void update();
	virtual void sdlEvent(SDL_Event &event);
};
