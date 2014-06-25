#pragma once

#include "HydroGPU/Solver.h"
#include "CLApp/CLApp.h"
#include "LuaCxx/State.h"
#include "LuaCxx/GlobalTable.h"
#include "LuaCxx/Ref.h"
#include "Tensor/Vector.h"

struct HydroGPUApp : public ::CLApp::CLApp {
	typedef ::CLApp::CLApp Super;

	GLuint gradientTex;
	
	cl::ImageGL gradientTexMem;	//as it is written, data is read from this for mapping values to colors

	std::shared_ptr<Solver> solver;

	//config
	std::string configFilename;
	std::string configString;
	std::string solverName;
	int dim;
	cl_int4 size;
	real4 xmin, xmax;
	int doUpdate;	//0 = no, 1 = continuous, 2 = single step
	int maxFrames;	//run this far and pause.  -1 = forever = default
	int currentFrame;
	bool useFixedDT;
	real fixedDT;
	real cfl;
	int displayMethod;
	float displayScale;
	Tensor::Vector<int,3> boundaryMethods;
	bool useGravity;
	double gamma;
	LuaCxx::State lua;
	real4 dx;
	
	//input
	bool leftButtonDown;
	bool rightButtonDown;
	bool leftShiftDown;
	bool rightShiftDown;
	bool leftGuiDown;
	bool rightGuiDown;
	
	//display
	Tensor::Vector<int,2> screenSize;
	float aspectRatio;

	bool showTimestep;

	HydroGPUApp();

	virtual int main(const std::vector<std::string>& args);
	virtual void init();
	virtual void shutdown();
	virtual void resetState();
	virtual void resize(int width, int height);
	virtual void update();
	virtual void sdlEvent(SDL_Event &event);
};

inline std::ostream& operator<<(std::ostream& o, real4 v) {
	return o << v.s[0] << ", " << v.s[1] << ", " << v.s[2] << ", " << v.s[3];
}

inline std::ostream& operator<<(std::ostream& o, cl_int4 v) {
	return o << v.s[0] << ", " << v.s[1] << ", " << v.s[2] << ", " << v.s[3];
}

inline std::ostream& operator<<(std::ostream& o, cl::NDRange &range) {
	o << "(";
	const char *comma = "";
	for (int i = 0; i < range.dimensions(); ++i) {
		o << comma << range[i];
		comma = ", ";
	}
	return o << ")";
}

