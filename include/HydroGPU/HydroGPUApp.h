#pragma once

#include "HydroGPU/RoeSolver.h"
#include "CLApp/CLApp.h"
#include "Tensor/Vector.h"

struct HydroGPUApp : public ::CLApp::CLApp {
	typedef ::CLApp::CLApp Super;

	GLuint fluidTex;
	GLuint gradientTex;
	
	cl_mem fluidTexMem;		//data is written to this buffer before rendering
	cl_mem gradientTexMem;	//as it is written, data is read from this for mapping values to colors

	std::shared_ptr<RoeSolver> solver;

	cl_int2 size;
	  
	bool leftButtonDown;
	bool rightButtonDown;
	bool leftShiftDown;
	bool rightShiftDown;
	bool leftGuiDown;
	bool rightGuiDown;
	Tensor::Vector<real,2> mousePos, mouseVel;
	Tensor::Vector<real,2> xmin, xmax;
	int doUpdate;	//0 = no, 1 = continuous, 2 = single step
	Tensor::Vector<int,2> screenSize;
	float viewZoom;
	Tensor::Vector<float,2> viewPos;
	float aspectRatio;

	HydroGPUApp();

	virtual int main(std::vector<std::string> args);
	virtual void init();
	virtual void shutdown();
	virtual void resize(int width, int height);
	virtual void update();
	virtual void sdlEvent(SDL_Event &event);
};
