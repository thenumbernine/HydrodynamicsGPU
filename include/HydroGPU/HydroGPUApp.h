#pragma once

#include "HydroGPU/Shared/Common.h"
#include "GLApp/GLApp.h"
#include "GLApp/ViewBehavior.h"
#include "LuaCxx/State.h"
#include "LuaCxx/GlobalTable.h"
#include "LuaCxx/Ref.h"
#include "GLCxx/Texture.h"
#include "GLCxx/gl.h"
#include "Tensor/Tensor.h"
#include "CLCommon/CLCommon.h"

#include <map>

namespace GLApp {
struct View;
struct ViewOrtho;
struct ViewFrustum;
}

namespace ImGuiCommon {
struct ImGuiCommon;
}

namespace HydroGPU {

namespace Plot {
struct Plot;
struct Graph;
struct HeatMap;
struct Iso3D;
struct VectorField;
}

namespace Solver {
struct Solver;
}

struct HydroGPUApp : public ::GLApp::ViewBehavior<::GLApp::GLApp> {
	using Super = ::GLApp::ViewBehavior<::GLApp::GLApp>;

	std::shared_ptr<CLCommon::CLCommon> clCommon;
	bool hasGLSharing = {};
	bool hasFP64 = {};
	
	std::shared_ptr<ImGuiCommon::ImGuiCommon> gui;

	GLCxx::Texture gradientTex;

	int equationIndex = {};

	std::map<std::string, std::vector<std::string>> initCondNamesForEqns;
	int initCondIndex = {};

	//solverGensForEqns[index corresponding with pair whose first is the equation name][index corresponding with pair whose first is the solver name] = function to create solver
	using SolverPtr = std::shared_ptr<Solver::Solver>;
	using SolverGenFunc = std::function<SolverPtr()>;
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
	
	int solverForEqnIndex = {};
	
	SolverPtr solver;

	//config
	std::string configFilename = "config.lua";
	std::string configString;
	std::string solverName = "EulerBurgers";
	int dim = 0;
	cl_int4 size = {0,0,0,0};
	real4 xmin, xmax;
	int doUpdate = 1;	//0 = no, 1 = continuous, 2 = single step
	int maxFrames = -1;	//run this far and pause.  -1 = forever = default
	int currentFrame = -1;
	bool useFixedDT = false;
	real fixedDT = .001;
	float cfl = .5;
	bool showHeatMap = true;
	bool showIso3D = true;
	Tensor::tensor<int, 3, 2> boundaryMethods;
	bool useGravity = false;
	int gaussSeidelMaxIter = 20;	//max iterations for Gauss-Seidel max iterations
	LuaCxx::State lua;
	real4 dx;
	bool showVectorField = true;
	bool createAnimation = false;
	bool showTimestep = false;

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

	virtual ~HydroGPUApp();
	virtual void init(const Init& args);
	virtual void onUpdate();
	virtual void onSDLEvent(SDL_Event &event);
	virtual std::string getTitle() { return "Hydrodynmics on the GPU"; }
};

inline std::ostream& operator<<(std::ostream& o, const real4& v) {
	return o << v.s[0] << ", " << v.s[1] << ", " << v.s[2] << ", " << v.s[3];
}

inline std::ostream& operator<<(std::ostream& o, const cl_int4& v) {
	return o << v.s[0] << ", " << v.s[1] << ", " << v.s[2] << ", " << v.s[3];
}

inline std::ostream& operator<<(std::ostream& o, const cl::NDRange &range) {
	o << "(";
	const char *comma = "";
	for (int i = 0; i < (int)range.dimensions(); ++i) {
		o << comma << range[i];
		comma = ", ";
	}
	return o << ")";
}

}
