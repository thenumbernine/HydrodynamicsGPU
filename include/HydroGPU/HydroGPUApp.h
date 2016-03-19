#pragma once

#include "LuaCxx/State.h"
#include "LuaCxx/GlobalTable.h"
#include "LuaCxx/Ref.h"
#include "GLApp/GLApp.h"
#include "Tensor/Tensor.h"
#include "CLCommon/CLCommon.h"
#include "HydroGPU/Shared/Common.h"	//real4
#include <map>

namespace ImGUICommon {
struct ImGUICommon;
}

namespace HydroGPU {

namespace Plot {
struct Camera;
struct CameraOrtho;
struct CameraFrustum;
struct Plot;
struct VectorField;
struct Graph;
}

namespace Solver {
struct Solver;
}

struct HydroGPUApp : public ::GLApp::GLApp {
	typedef ::GLApp::GLApp Super;

	std::shared_ptr<CLCommon::CLCommon> clCommon;
	std::shared_ptr<ImGUICommon::ImGUICommon> gui;

	GLuint gradientTex;
	
	cl::ImageGL gradientTexMem;	//as it is written, data is read from this for mapping values to colors

	std::vector<std::string> equationNames;
	std::string equationNamesSeparated;
	int equationIndex;
	
	std::vector<std::pair<std::string, std::vector<std::pair<std::string, std::function<std::shared_ptr<Solver::Solver>()>>>>> solverGensForEqns;
	std::map<std::string, std::string> solverNamesSeparatedForEqn;
	int solverForEqnIndex;
	
	std::vector<std::pair<std::string, std::function<std::shared_ptr<Solver::Solver>()>>> solverGens;
	
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
	bool showHeatMap;
	int heatMapVariable;	//TODO the enumeration of these values is dependent on the solver equation 
	float heatMapColorScale;
	Tensor::Tensor<int, Tensor::Lower<3>, Tensor::Lower<2>> boundaryMethods;
	bool useGravity;
	int gaussSeidelMaxIter;	//max iterations for Gauss-Seidel max iterations
	LuaCxx::State lua;
	real4 dx;
	bool showVectorField;
	float vectorFieldScale;
	bool createAnimation;
	bool showTimestep;
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
	
	//construct this after the program has been compiled
	std::shared_ptr<HydroGPU::Plot::VectorField> vectorField;
	std::shared_ptr<HydroGPU::Plot::Plot> plot;
	std::shared_ptr<HydroGPU::Plot::Graph> graph;
	std::shared_ptr<HydroGPU::Plot::Camera> camera;
	std::shared_ptr<HydroGPU::Plot::CameraFrustum> cameraFrustum;
	std::shared_ptr<HydroGPU::Plot::CameraOrtho> cameraOrtho;

	HydroGPUApp();

	virtual int main(const std::vector<std::string>& args);
	virtual void init();
	virtual void shutdown();
	virtual void resize(int width, int height);
	virtual void update();
	virtual void sdlEvent(SDL_Event &event);
protected:
	void createPlot();
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
