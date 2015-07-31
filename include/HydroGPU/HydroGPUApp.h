#pragma once

#include "HydroGPU/Solver/Solver.h"
#include "GLApp/GLApp.h"
#include "CLCommon/CLCommon.h"
#include "LuaCxx/State.h"
#include "LuaCxx/GlobalTable.h"
#include "LuaCxx/Ref.h"
#include "Tensor/Tensor.h"

namespace HydroGPU {
namespace Plot {
struct Camera;
struct Plot;
struct VectorField;
struct Graph;
}

struct HydroGPUApp : public ::GLApp::GLApp {
	typedef ::GLApp::GLApp Super;

	std::shared_ptr<CLCommon::CLCommon> clCommon;

	GLuint gradientTex;
	
	cl::ImageGL gradientTexMem;	//as it is written, data is read from this for mapping values to colors

	std::shared_ptr<HydroGPU::Solver::Solver> solver;

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
	int heatMapVariable;	//TODO the enumeration of these values is dependent on the solver equation 
	float heatMapColorScale;
	Tensor::Tensor<int, Tensor::Lower<3>, Tensor::Lower<2>> boundaryMethods;
	bool useGravity;
	int gaussSeidelMaxIter;	//max iterations for Gauss-Seidel max iterations
	LuaCxx::State lua;
	real4 dx;
	bool showVectorField;
	float vectorFieldScale;
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
	
	//construct this after the program has been compiled
	std::shared_ptr<HydroGPU::Plot::VectorField> vectorField;
	std::shared_ptr<HydroGPU::Plot::Plot> plot;
	std::shared_ptr<HydroGPU::Plot::Graph> graph;
	std::shared_ptr<HydroGPU::Plot::Camera> camera;

	HydroGPUApp();

	virtual int main(const std::vector<std::string>& args);
	virtual void init();
	virtual void shutdown();
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

}

