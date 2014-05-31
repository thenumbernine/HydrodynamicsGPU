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

	Vector<int,3> size;
	  
	bool leftButtonDown;
	bool rightButtonDown;
	bool leftShiftDown;
	bool rightShiftDown;
	bool leftGuiDown;
	bool rightGuiDown;
	Vector<real,2> mousePos, mouseVel;
	Vector<real,2> xmin, xmax;
	int doUpdate;	//0 = no, 1 = continuous, 2 = single step
	Vector<int,2> screenSize;
	float viewZoom;
	Vector<float,2> viewPos;
	float aspectRatio;

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
