#pragma once

#include "LuaCxx/State.h"
#include "LuaCxx/GlobalTable.h"
#include "LuaCxx/Ref.h"
#include "GLApp/GLApp.h"
#include "Tensor/Tensor.h"
#include "CLCommon/CLCommon.h"
#include "HydroGPU/Shared/Common.h"	//real4
#include "Common/gl.h"

#include <map>

namespace ImGuiCommon {
struct ImGuiCommon;
}

namespace HydroGPU {

namespace Plot {
struct Camera;
struct CameraOrtho;
struct CameraFrustum;
struct Plot;
struct Graph;
struct HeatMap;
struct Iso3D;
struct VectorField;
}

namespace Solver {
struct Solver;
}

struct HydroGPUApp : public ::GLApp::GLApp {
	typedef ::GLApp::GLApp Super;

	std::shared_ptr<CLCommon::CLCommon> clCommon;
	bool hasGLSharing;
	bool hasFP64;
	
	std::shared_ptr<ImGuiCommon::ImGuiCommon> gui;

	GLuint gradientTex;
	
	cl::ImageGL gradientTexMem;	//as it is written, data is read from this for mapping values to colors

	int equationIndex;

	std::map<std::string, std::vector<std::string>> initCondNamesForEqns;
	int initCondIndex;

	//solverGensForEqns[index corresponding with pair whose first is the equation name][index corresponding with pair whose first is the solver name] = function to create solver
	typedef std::shared_ptr<Solver::Solver> SolverPtr;
	typedef std::function<SolverPtr()> SolverGenFunc;
	struct SolverGenPair {
		SolverGenPair(const std::string& name_, SolverGenFunc func_)
		: name(name_), func(func_) {}
		std::string name;
		SolverGenFunc func;
	};
	struct SolverEqnsPair {
		SolverEqnsPair(const std::string& name_, const std::vector<SolverGenPair>& generators_)
		: name(name_), generators(generators_) {}
		std::string name;
		std::vector<SolverGenPair> generators;
	};
	std::vector<SolverEqnsPair> solverGensForEqns;
	
	int solverForEqnIndex;
	
	SolverPtr solver;

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
	float cfl;
	bool showHeatMap;
	bool showIso3D;
	Tensor::Tensor<int, Tensor::Lower<3>, Tensor::Lower<2>> boundaryMethods;
	bool useGravity;
	int gaussSeidelMaxIter;	//max iterations for Gauss-Seidel max iterations
	LuaCxx::State lua;
	real4 dx;
	bool showVectorField;
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
	
	//screenshots, converting cl mem to textures
	std::shared_ptr<HydroGPU::Plot::Plot> plot;
	
	//1D and 2D heightmap graphs
	std::shared_ptr<HydroGPU::Plot::Graph> graph;
	
	//2D heat maps ... possibly 3D slices someday?
	std::shared_ptr<HydroGPU::Plot::HeatMap> heatMap;
	
	//3D isobars
	std::shared_ptr<HydroGPU::Plot::Iso3D> iso3D;
	
	//1D 2D and 3D vector fields
	std::shared_ptr<HydroGPU::Plot::VectorField> vectorField;

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
	for (int i = 0; i < (int)range.dimensions(); ++i) {
		o << comma << range[i];
		comma = ", ";
	}
	return o << ")";
}

}
