#pragma once

#include "HydroGPU/RoeSolver.h"
#include "HydroGPU/CLApp.h"
#include "TensorMath/Vector.h"

struct HydroGPUApp : public CLApp {
	typedef CLApp Super;

	GLuint fluidTex;
	GLuint gradientTex;
	
	cl_mem fluidTexMem;		//data is written to this buffer before rendering
	cl_mem gradientTexMem;	//as it is written, data is read from this for mapping values to colors

	Solver *solver;

	int dim;	//1,2 or 3
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

	virtual int main(std::vector<std::string> args);
	virtual void init();
	virtual void shutdown();
	virtual void resize(int width, int height);
	virtual void update();
	virtual void sdlEvent(SDL_Event &event);
};
