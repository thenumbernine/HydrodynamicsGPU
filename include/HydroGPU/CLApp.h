#pragma once

#include "GLApp/GLApp.h"
#include <OpenCL/cl.hpp>

struct CLApp : public GLApp {
	bool useGPU;	//whether we want to request the GPU or CPU.  required prior to init()
	cl::Platform platform;
	cl::Device device;
	cl::Context context;
	cl::CommandQueue commands;

	CLApp();
	cl::Platform getPlatform();
	cl::Device getDevice(cl::Platform platform);
	
	virtual void init();
};

