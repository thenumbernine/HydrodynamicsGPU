#include "HydroGPU/Solver/EulerHLL.h"
#include "HydroGPU/Solver/EulerHLLC.h"
#include "HydroGPU/Solver/EulerBurgers.h"
#include "HydroGPU/Solver/EulerRoe.h"
#include "HydroGPU/Solver/SRHDRoe.h"
#include "HydroGPU/Solver/MHDBurgers.h"
#include "HydroGPU/Solver/MHDHLLC.h"
#include "HydroGPU/Solver/MHDRoe.h"
#include "HydroGPU/Solver/MaxwellRoe.h"
#include "HydroGPU/Solver/ADM1DRoe.h"
#include "HydroGPU/Solver/ADM3DRoe.h"
#include "HydroGPU/Solver/BSSNOKRoe.h"
#include "HydroGPU/Equation/Euler.h"
#include "HydroGPU/Equation/SRHD.h"
#include "HydroGPU/Equation/MHD.h"
#include "HydroGPU/Equation/Maxwell.h"
#include "HydroGPU/Equation/ADM1D.h"
#include "HydroGPU/Equation/ADM2DSpherical.h"
#include "HydroGPU/Equation/ADM3D.h"
#include "HydroGPU/Equation/BSSNOK.h"
#include "HydroGPU/Plot/Graph.h"
#include "HydroGPU/Plot/HeatMap.h"
#include "HydroGPU/Plot/Iso3D.h"
#include "HydroGPU/Plot/Plot.h"
#include "HydroGPU/Plot/VectorField.h"
#include "HydroGPU/HydroGPUApp.h"
#include "ImGuiCommon/ImGuiCommon.h"
#include "Profiler/Profiler.h"
#include "CLCommon/cl.hpp"
#include "GLApp/ViewOrtho.h"
#include "GLApp/ViewFrustum.h"
#include "GLCxx/gl.h"
#include "Common/Exception.h"
#include "Common/File.h"
#include "Common/Macros.h"
#include "SDL.h"
#include <iostream>
#include <algorithm>

//Sooo ... ImGuiCommon is now using cimgui.h.
//And I was using ImGui here, which is the proper C++ thing to do.
//However, now that I'm already including cimgui.h via ImGuiCommon, it turns out cimgui.h and imgui.h cause conflicts when included together.
//So I need to pick just one and stick with it.
//And for now that will be imgui.h
//#include "imgui.h"

static std::vector<const char*> getCStrsFromStrVector(const std::vector<std::string>& v) {
	std::vector<const char*> result;
	for (const std::string& s : v) {
		result.push_back(s.c_str());
	}
	return result;
}

namespace HydroGPU {

void HydroGPUApp::init(const Init& args) {
	Super::init(args);

	for (int i = 1; i < (int)args.size(); ++i) {
		if (i < (int)args.size()-1 && args[i] == "-e") {
			configString = args[++i];
		} else {
			configFilename = args[i];
		}
	}

#define MAKE_SOLVER(solver) SolverGenPair(#solver, [this]()->SolverPtr{ return std::make_shared<Solver::solver>(this); })
	solverGensForEqns = {
		SolverEqnsPair("Euler", {
			MAKE_SOLVER(EulerBurgers),
			MAKE_SOLVER(EulerHLL),
			MAKE_SOLVER(EulerHLLC),
			MAKE_SOLVER(EulerRoe),
		}),
		SolverEqnsPair("SRHD", {
			MAKE_SOLVER(SRHDRoe),
		}),
		SolverEqnsPair("MHD", {
			MAKE_SOLVER(MHDBurgers),
			MAKE_SOLVER(MHDHLLC),
			MAKE_SOLVER(MHDRoe),
		}),
		SolverEqnsPair("Maxwell", {
			MAKE_SOLVER(MaxwellRoe),
		}),
		SolverEqnsPair("ADM1D", {
			MAKE_SOLVER(ADM1DRoe),
		}),
		SolverEqnsPair("ADM3D", {
			MAKE_SOLVER(ADM3DRoe),
		}),
		SolverEqnsPair("BSSNOK", {
			MAKE_SOLVER(BSSNOKRoe),
		}),
	};
#undef MAKE_SOLVER

	for (int i = 0; i < 4; ++i) {
		size.s[i] = 1;	//default each dimension to a point.  so if lua doesn't define it then the dimension will be ignored
		xmin.s[i] = -.5f;
		xmax.s[i] = .5f;
	}

	{
		std::string extensionStr = (char*)glGetString(GL_EXTENSIONS);
		std::istringstream iss(extensionStr);
		std::vector<std::string> extensions;
		std::copy(std::istream_iterator<std::string>(iss), std::istream_iterator<std::string>(), std::back_inserter<std::vector<std::string>>(extensions));
		std::sort(extensions.begin(), extensions.end());
		std::cout << "GL_EXTENSIONS:" << std::endl;
		for (std::string &s : extensions) {
			std::cout << "\t" << s << std::endl;
		}
		std::cout << std::endl;
	}

	//config before Super::init so we can provide it 'useGPU'
	std::cout << "loading config file " << configFilename << " ..." << std::endl;
	lua.loadFile(configFilename);
	if (!configString.empty()) {
		std::cout << "loading config string " << configString << std::endl;
		lua.loadString(configString);
	}
	std::cout << "...loaded config file" << std::endl;

	bool useGPU = true;
	lua["useGPU"] >> useGPU;
	for (int i = 0; i < 3; ++i) {
		if (!lua["size"].isNil()) lua["size"][i+1] >> size.s[i];
		if (!lua["xmin"].isNil()) lua["xmin"][i+1] >> xmin.s[i];
		if (!lua["xmax"].isNil()) lua["xmax"][i+1] >> xmax.s[i];
	}

	std::vector<std::vector<std::string>> boundaryMethodNames(3);
	for (int i = 0; i < 3; ++i) {
		boundaryMethodNames[i].resize(2);
		if (!lua["boundaryMethods"].isNil()) {
			if (lua["boundaryMethods"][i+1].isTable()) {
				lua["boundaryMethods"][i+1]["min"] >> boundaryMethodNames[i][0];
				lua["boundaryMethods"][i+1]["max"] >> boundaryMethodNames[i][1];
			}
		}
	}

	for (const SolverEqnsPair& p : solverGensForEqns) {
		const std::string& eqnName = p.name;
		std::vector<std::string> initCondNames;
		for (int i = 1; i <= lua["initConds"].len(); ++i) {
			std::string initCondName;
			lua["initConds"][i]["name"] >> initCondName;
			for (int j = 1; j <= lua["initConds"][i]["equations"].len(); ++j) {
				std::string initCondEqnName;
				if ((lua["initConds"][i]["equations"][j] >> initCondEqnName).good() &&
					initCondEqnName == eqnName)
				{
					initCondNames.push_back(initCondName);
					break;
				}
			}
		}
		initCondNamesForEqns[eqnName] = initCondNames;
	}

	lua["maxFrames"] >> maxFrames;
	lua["showTimestep"] >> showTimestep;
	lua["useFixedDT"] >> useFixedDT;
	lua["fixedDT"] >> fixedDT;
	lua["cfl"] >> cfl;
	lua["useGravity"] >> useGravity;
	lua["gaussSeidelMaxIter"] >> gaussSeidelMaxIter;

	bool disableGUI = false;
	lua["disableGUI"] >> disableGUI;

	//store dimension as last non-1 size
	for (dim = 3; dim > 0; --dim) {
		if (size.s[dim-1] > 1) {
			break;
		} if (size.s[dim-1] != 1) {
			throw Common::Exception() << "size[" << dim << "] has invalid value of " << size.s[dim-1];
		}
	}
	if (dim == 0) throw Common::Exception() << "couldn't find any size information";
	std::cout << "dim " << dim << std::endl;

	for (int i = 0; i < 3; ++i) {
		dx.s[i] = (xmax.s[i] - xmin.s[i]) / (float)size.s[i];
	}
	std::cout << "xmin " << xmin << std::endl;
	std::cout << "xmax " << xmax << std::endl;
	std::cout << "size " << size << std::endl;
	std::cout << "dx " << dx << std::endl;

	if (!disableGUI) {
		gui = std::make_shared<ImGuiCommon::ImGuiCommon>(window, context);
	}

	//TODO put this in CLCommon, where a similar function operating on vectors exists
	auto checkHasGLSharing = [](const cl::Device& device)-> bool {
		std::vector<std::string> extensions = CLCommon::getExtensions(device);
		return std::find(extensions.begin(), extensions.end(), "cl_khr_gl_sharing") != extensions.end()
			|| std::find(extensions.begin(), extensions.end(), "cl_APPLE_gl_sharing") != extensions.end();
	};

	auto checkHasFP64 = [](const cl::Device& device)-> bool {
		std::vector<std::string> extensions = CLCommon::getExtensions(device);
		return std::find(extensions.begin(), extensions.end(), "cl_khr_fp64") != extensions.end();
	};

	clCommon = std::make_shared<CLCommon::CLCommon>(
		useGPU,
		/*verbose=*/true,
		/*pickDevice=*/[&](const std::vector<cl::Device>& devices_) -> std::vector<cl::Device>::const_iterator {
			
			//sort with a preference to sharing
			std::vector<cl::Device> devices = devices_;
			std::sort(
				devices.begin(),
				devices.end(),
				[&](const cl::Device& a, const cl::Device& b) -> bool {
					return 
						//has GL sharing is most important, has fp64 is next
						//TODO let the config file pick what features to request
						(2*checkHasGLSharing(a) + checkHasFP64(a)) 
						> (2*checkHasGLSharing(b) + checkHasFP64(b));
				});

			cl::Device best = devices[0];
			//return std::find<std::vector<cl::Device>::const_iterator, cl::Device>(devices_.begin(), devices_.end(), best);
			for (std::vector<cl::Device>::const_iterator iter = devices_.begin(); iter != devices_.end(); ++iter) {
				if ((*iter)() == best()) return iter;
			}
			throw Common::Exception() << "couldn't find a device";
		});
	
	hasGLSharing = checkHasGLSharing(clCommon->device);
std::cout << "hasGLSharing " << hasGLSharing << std::endl; 

	//override if it exists
	lua["hasGLSharing"] >> hasGLSharing;

	hasFP64 = checkHasFP64(clCommon->device);
std::cout << "hasFP64 " << hasFP64 << std::endl;

	glEnable(GL_DEPTH_TEST);

	//gradient texture
	{
		const int width = 256;
		unsigned char data[width*3];
//TODO texture options for heat map vs isobar display
#if 1	//color gradient
		float colors[][3] = {
			{0, 0, 0},
			{0, 0, 1},
			{0, 1, 1},
			{1, 1, 0},
			{1, 0, 0}
		};

		for (int i = 0; i < width; ++i) {
			float f = (float)i / (float)width * (float)numberof(colors);
			int ci = (int)f;
			int ci2 = (ci + 1) % numberof(colors);
			float s = f - (float)ci;
			if (ci >= (int)numberof(colors)) {
				ci = numberof(colors)-1;
				s = 0;
			}

			for (int j = 0; j < 3; ++j) {
				data[3 * i + j] = (unsigned char)(255. * (colors[ci][j] * (1.f - s) + colors[ci2][j] * s));
			}
		}
#endif
#if 0	//isobars
		memset(data, 0, sizeof(data));
		memset(data, -1, sizeof(data)/8);
#endif
		gradientTex = GLCxx::Texture1D()
			.bind()
			.create1D(width, GL_RGB, GL_RGB, GL_UNSIGNED_BYTE, data)
			//.generateMipmap()
			//.setParam<GL_TEXTURE_MIN_FILTER>(GL_LINEAR_MIPMAP_LINEAR)	//isobars look good with mipmapping
			.setParam<GL_TEXTURE_MIN_FILTER>(GL_NEAREST)	//heatmaps do not
			.setParam<GL_TEXTURE_MAG_FILTER>(GL_LINEAR)
			.unbind();
	}

	lua["solverName"] >> solverName;
	std::cout << "solverName " << solverName << std::endl;
	{
		bool found = false;
		for (const SolverEqnsPair &p : solverGensForEqns) {
			for (const SolverGenPair &q : p.generators) {
				if (q.name == solverName) {
					solver = q.func();
					solver->init();	//...now that the vtable is in place
					solver->resetState();
					found = true;
					break;
				}
			}
			if (found) break;
		}
		if (!found) {
			throw Common::Exception() << "unknown solver " << solverName;
		}
	}

	//find the index of the equation associated with the selected solver (for the combo box)
	
	for (equationIndex = 0; equationIndex < (int)solverGensForEqns.size(); ++equationIndex) {
		if (solver->getEquation()->name() == solverGensForEqns[equationIndex].name) break;
	}
	if (equationIndex == (int)solverGensForEqns.size()) equationIndex = 0;

	//find the solver index in the list of solvers for this equation
	for (solverForEqnIndex = 0; solverForEqnIndex < (int)solverGensForEqns[equationIndex].generators.size(); ++solverForEqnIndex) {
		if (solver->name() == solverGensForEqns[equationIndex].generators[solverForEqnIndex].name) break;
	}
	if (solverForEqnIndex == (int)solverGensForEqns[equationIndex].generators.size()) solverForEqnIndex = 0;

	{
		std::string initCondName;
		if ((lua["initCondName"] >> initCondName).good()) {
			std::string equationName = solverGensForEqns[equationIndex].name;
			std::vector<std::string>& initCondNames = initCondNamesForEqns[equationName];
			initCondIndex = std::find(initCondNames.begin(), initCondNames.end(), initCondName) - initCondNames.begin();
		}
	}

	//needs solver->program to be created
	int vectorFieldResolution = 16;
	float vectorFieldScale = .125f;
	int vectorFieldVariable = 0;
	if (!lua["vectorField"].isNil()) {
		lua["vectorField"]["enabled"] >> showVectorField;
		lua["vectorField"]["scale"] >> vectorFieldScale;
		lua["vectorField"]["resolution"] >> vectorFieldResolution;
	
		std::string variableName;
		if ((lua["vectorField"]["variable"] >> variableName).good()) {
			vectorFieldVariable = std::find(
				solver->getEquation()->vectorFieldVars.begin(),
				solver->getEquation()->vectorFieldVars.end(),
				variableName)
				- solver->getEquation()->vectorFieldVars.begin();
			if (vectorFieldVariable == (int)solver->getEquation()->vectorFieldVars.size()) {
				vectorFieldVariable = 0;
				std::cerr << "couldn't interpret display method " << variableName << std::endl;
			}
		}
	}
	vectorField = std::make_shared<Plot::VectorField>(solver, vectorFieldResolution);
	vectorField->scale = vectorFieldScale;
	vectorField->variable = vectorFieldVariable;

	if (lua["camera"].isNil()) {
		throw Common::Exception() << "unknown camera";
	}
	
	if (!lua["camera"]["pos"].isNil()) {
		lua["camera"]["pos"][1] >> viewFrustum->pos(0);
		lua["camera"]["pos"][2] >> viewFrustum->pos(1);
		lua["camera"]["pos"][3] >> viewFrustum->pos(2);
	}
	if (!lua["camera"]["orbit"].isNil()) {
		lua["camera"]["orbit"][1] >> viewFrustum->orbit(0);
		lua["camera"]["orbit"][2] >> viewFrustum->orbit(1);
		lua["camera"]["orbit"][3] >> viewFrustum->orbit(2);
	}
	if (!lua["camera"]["pos"].isNil()) {
		lua["camera"]["pos"][1] >> viewOrtho->pos(0);
		lua["camera"]["pos"][2] >> viewOrtho->pos(1);
	}
	lua["camera"]["zoom"] >> viewOrtho->zoom(0);
	viewOrtho->zoom(1) = viewOrtho->zoom(0);

	{
		std::string mode;
		lua["camera"]["mode"] >> mode;
		if (mode == "ortho") {
			view = viewOrtho;
		} else if (mode == "frustum") {
			view = viewFrustum;
		}
	}

	//needs solver->program to be created
	plot = std::make_shared<Plot::Plot>(this);
	plot->init();

	if (dim == 1 || dim == 2) {
		graph = std::make_shared<Plot::Graph>(this);
		{
			std::vector<std::string> graphVariableNames;
			LuaCxx::Ref graphVariablesRef = lua["graph"]["variables"];
			if (graphVariablesRef.isTable()) {
				for (LuaCxx::Ref::iterator i = graphVariablesRef.begin(); i != graphVariablesRef.end(); ++i) {
					std::string varName = (std::string)i.value;	//TODO operator-> for i.value?
					graphVariableNames.push_back(varName);
				}
			} else if (graphVariablesRef.isNil()) {
				//by default add all
				graphVariableNames = solver->getEquation()->displayVariables;
			}

			//enable all graph variables that are in graph.variables Lua table
			for (std::vector<std::string>::const_iterator i = graphVariableNames.begin(); i != graphVariableNames.end(); ++i) {
				for (std::vector<HydroGPU::Plot::Graph::Variable>::iterator j = graph->variables.begin(); j != graph->variables.end(); ++j) {
					if (*i == j->name) {
						j->enabled = true;
						break;
					}
				}
			}
		}
	}

	//3D heatmap is a pointcloud 
	if (dim == 2) {// || dim == 3) {
		heatMap = std::make_shared<Plot::HeatMap>(this);
		if (!lua["heatMap"].isNil()) {
			lua["heatMap"]["enabled"] >> showHeatMap;
			lua["heatMap"]["colorScale"] >> heatMap->scale;
			lua["heatMap"]["useLog"] >> heatMap->useLog;
			lua["heatMap"]["alpha"] >> heatMap->alpha;

			std::string variableName;
			if ((lua["heatMap"]["variable"] >> variableName).good()) {
				heatMap->variable = std::find(
					solver->getEquation()->displayVariables.begin(),
					solver->getEquation()->displayVariables.end(),
					variableName)
					- solver->getEquation()->displayVariables.begin();
				if (heatMap->variable == (int)solver->getEquation()->displayVariables.size()) {
					heatMap->variable = 0;
					std::cerr << "couldn't interpret display method " << variableName << std::endl;
				}
			}
		}
	}

	//3D isosurface is raytraced
	//TODO add in a marching cube display?
	if (dim == 3) {
		iso3D = std::make_shared<Plot::Iso3D>(this);
		if (!lua["iso3D"].isNil()) {
			lua["iso3D"]["enabled"] >> showIso3D;
			lua["iso3D"]["colorScale"] >> iso3D->scale;
			lua["iso3D"]["useLog"] >> iso3D->useLog;
			lua["iso3D"]["alpha"] >> iso3D->alpha;
			
			std::string variableName;
			if ((lua["iso3D"]["variable"] >> variableName).good()) {
				iso3D->variable = std::find(
					solver->getEquation()->displayVariables.begin(),
					solver->getEquation()->displayVariables.end(),
					variableName)
					- solver->getEquation()->displayVariables.begin();
				if (iso3D->variable == (int)solver->getEquation()->displayVariables.size()) {
					iso3D->variable = 0;
					std::cerr << "couldn't interpret display method " << variableName << std::endl;
				}
			}		
		}
	}

	for (int i = 0; i < 3; ++i) {
		for (int minmax = 0; minmax < 2; ++minmax) {
			if (boundaryMethodNames[i][minmax].empty()) continue;
			std::vector<std::string>& equationBoundaryMethods = solver->getEquation()->boundaryMethods;
			std::vector<std::string>::iterator iter = std::find(equationBoundaryMethods.begin(), equationBoundaryMethods.end(), boundaryMethodNames[i][minmax]);
			boundaryMethods(i,minmax) = 
				(iter == equationBoundaryMethods.end()) 
				? (int)equationBoundaryMethods.size() 
				: (iter - equationBoundaryMethods.begin());
			if (boundaryMethods(i,minmax) == (int)equationBoundaryMethods.size()) {
				//special case
				if (boundaryMethodNames[i][minmax] == "NONE") {
					boundaryMethods(i,minmax) = -1;
				} else {
					throw Common::Exception() << "couldn't interpret boundary method " << boundaryMethodNames[i][minmax];
				}
			}
		}
	}

	int err = glGetError();
	if (err) throw Common::Exception() << "GL error " << err;

	std::cout << "Success!" << std::endl;
}

HydroGPUApp::~HydroGPUApp() {
	clCommon.reset();
}

void HydroGPUApp::onUpdate() {
PROFILE_BEGIN_FRAME()

	++currentFrame;
	if (currentFrame == maxFrames) {
		doUpdate = 0;
	}

	Super::onUpdate();	//glClear, view->setup

	bool guiDown = leftGuiDown || rightGuiDown;
	if (mouse.rightDown || (mouse.leftDown && guiDown)) {
		//TODO - direct edit of the field ... solver->addDrop();
	}

	if (doUpdate) {
		solver->update();
		if (doUpdate == 2) doUpdate = 0;
	}

	//no point in showing the graph in ortho
	if (graph) graph->display();

	glDepthMask(GL_FALSE);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE);

	if (showHeatMap && heatMap) heatMap->display();
	if (showIso3D && iso3D) iso3D->display();
	if (showVectorField) vectorField->display();
	
	glDisable(GL_BLEND);
	glDepthMask(GL_TRUE);

	//do this before rendering the gui so it stays out of the picture
	if (createAnimation) plot->screenshot();

	/*
	What should be customizable?
	- a Lua interface into everything would be nice ... if we were using LuaJIT
	... then *everything* C-callable would be accessible immediately
	*/
	if (gui) gui->onUpdate([&](){
		//how do you change the window title from "Debug"?

		if (igCollapsingHeader_TreeNodeFlags("setup", ImGuiTreeNodeFlags_None)) {
			//list equations
			int lastEquationIndex = equationIndex;
			std::vector<const char*> equationNamesCStrs;
			for (const SolverEqnsPair& p : solverGensForEqns) { equationNamesCStrs.push_back(p.name.c_str()); } 
			igCombo_Str_arr("equation", &equationIndex, equationNamesCStrs.data(), equationNamesCStrs.size(), -1);
			if (lastEquationIndex != equationIndex) {

				//if we are changing equations then (most likely) our initial conditions will be invalid
				//(except cases where they belong to the same groups, I think, unless the resetState conversion works diferently,
				// either way, to be safe...)
				//we will have to reset the initial condition to something that certainly belongs to this equation
				initCondIndex = 0;
				std::string initCondName = initCondNamesForEqns[solverGensForEqns[equationIndex].name][initCondIndex];
				lua["initConds"][initCondName]["setup"]();

				solverForEqnIndex = 0;
				std::cout << "setting equation to " << solverGensForEqns[equationIndex].name << std::endl;
				solver = solverGensForEqns[equationIndex].generators[solverForEqnIndex].func();
				solver->init();
				//now we need an initial condition to match the new equation
				//but the initial conditions are provided in the script
				//this means (a) definining initial conditions somewhere in the script that the c++ code can see them
				// to enumerate and provide here in the gui
				// or (b) defining them in c++, which means faster constructor, but makes symmath more difficult to use
				solver->resetState();

				//regen aux things that depend on the solver
				plot = std::make_shared<Plot::Plot>(this);
				plot->init();
				
				std::shared_ptr<Plot::VectorField> newVectorField = std::make_shared<Plot::VectorField>(solver, vectorField->getResolution());
				newVectorField->scale = vectorField->scale;
				vectorField = newVectorField;
			
				if (graph && (dim == 1 || dim == 2)) {
					std::shared_ptr<Plot::Graph> newGraph = std::make_shared<Plot::Graph>(this);
					newGraph->variables = graph->variables;
					graph = newGraph;
				}
			}

			//list solvers of the equation
			int lastSolverForEqnIndex = solverForEqnIndex;
			std::vector<const char*> solverNamesForEqnCStrs;
			for (const SolverGenPair& p : solverGensForEqns[equationIndex].generators) {
				solverNamesForEqnCStrs.push_back(p.name.c_str());
			}
			igCombo_Str_arr("solver", &solverForEqnIndex, solverNamesForEqnCStrs.data(), solverNamesForEqnCStrs.size(), -1);
			if (lastSolverForEqnIndex != solverForEqnIndex) {
				std::cout << "setting solver to " << solverGensForEqns[equationIndex].generators[solverForEqnIndex].name << std::endl;
				SolverPtr newSolver = solverGensForEqns[equationIndex].generators[solverForEqnIndex].func();
				//but are there any solvers that have different #states for matching eqns?
				//until then ...

				newSolver->init();
				newSolver->resetState();

				//copy state memory *here*
				//if the states don't match then create a mapping according to the equations or something ...
				//TODO what about aux variables that need to be updated too?  like SRHD?
				//this is the same question that falls in line with the rk4 integrator push/pop modularity
				if (solver->numStates() == newSolver->numStates()) {
					size_t length = solver->numStates() * solver->getVolume();
					size_t bufferSize = sizeof(real) * length;
					clCommon->commands.enqueueCopyBuffer(solver->stateBuffer, newSolver->stateBuffer, 0, 0, bufferSize);
				}
				
				solver = newSolver;

				//regen aux things that depend on the solver
				plot = std::make_shared<Plot::Plot>(this);
				plot->init();
				
				std::shared_ptr<Plot::VectorField> newVectorField = std::make_shared<Plot::VectorField>(solver, vectorField->getResolution());
				newVectorField->scale = vectorField->scale;
				vectorField = newVectorField;

				if (graph && (dim == 1 || dim == 2)) {
					std::shared_ptr<Plot::Graph> newGraph = std::make_shared<Plot::Graph>(this);
					newGraph->variables = graph->variables;
					graph = newGraph;
				}
			}

			int lastInitCondIndex = initCondIndex;
			const std::string& eqnName = solverGensForEqns[equationIndex].name;
			std::vector<const char*> initCondNamesCStrs = getCStrsFromStrVector(initCondNamesForEqns[eqnName]);
			igCombo_Str_arr("init.cond.", &initCondIndex, initCondNamesCStrs.data(), initCondNamesCStrs.size(), -1);
			if (lastInitCondIndex != initCondIndex) {
				std::string initCondName = initCondNamesForEqns[solverGensForEqns[equationIndex].name][initCondIndex];
				std::cout << "setting up initial condition " << initCondName << std::endl;
				lua["initConds"][initCondName]["setup"]();
			}
		}

		if (igCollapsingHeader_TreeNodeFlags("boundaries", ImGuiTreeNodeFlags_None)) {
			std::vector<const char*> methodsCStrs = getCStrsFromStrVector(solver->getEquation()->boundaryMethods);
			methodsCStrs.insert(methodsCStrs.begin(), "NONE");
			for (int i = 0; i < dim; ++i) {
				for (int minmax = 0; minmax < 2; ++minmax) {
					std::stringstream comboTitle;
					comboTitle << (char)('x'+i) << " " << (minmax ? "max" : "min");
					++boundaryMethods(i,minmax);
					igCombo_Str_arr(comboTitle.str().c_str(), &boundaryMethods(i,minmax), methodsCStrs.data(), methodsCStrs.size(), -1);
					--boundaryMethods(i,minmax);
				}
			}
		}

		if (heatMap) {
			if (igCollapsingHeader_TreeNodeFlags("heat map", ImGuiTreeNodeFlags_None)) {
				igPushID_Str("heatMap");
				igCheckbox("show", &showHeatMap);

				std::vector<const char*> displayVariablesCStrs = getCStrsFromStrVector(solver->getEquation()->displayVariables);
				igCombo_Str_arr("variable ", &heatMap->variable, displayVariablesCStrs.data(), displayVariablesCStrs.size(), -1);

				igSliderFloat("scale", &heatMap->scale, 1e-10, 1e+10, "%.16f", ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_NoRoundToFormat);
			
				igCheckbox("log", &heatMap->useLog);
			
				igSliderFloat("alpha", &heatMap->alpha, 0.f, 1.f, "%.3f", ImGuiSliderFlags_None);
				igPopID();
			}
		}

		if (iso3D) {
			if (igCollapsingHeader_TreeNodeFlags("isosurfaces", ImGuiTreeNodeFlags_None)) {
				igPushID_Str("isosurfaces");
				igCheckbox("show ", &showIso3D);

				std::vector<const char*> displayVariablesCStrs = getCStrsFromStrVector(solver->getEquation()->displayVariables);
				igCombo_Str_arr("variable", &iso3D->variable, displayVariablesCStrs.data(), displayVariablesCStrs.size(), -1); 

				igSliderFloat("scale", &iso3D->scale, 1e-10, 1e+10, "%.16f", ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_NoRoundToFormat); 
			
				igCheckbox("log", &iso3D->useLog);
				
				igSliderFloat("alpha", &iso3D->alpha, 0.f, 1.f, "%.3f", ImGuiSliderFlags_None);
				igPopID();
			}
		}

		if (graph) {
			if (igCollapsingHeader_TreeNodeFlags("graph", ImGuiTreeNodeFlags_None)) {
				std::function<void(Plot::Graph::Variable&)> addVarControls = [&](Plot::Graph::Variable& var){
					const std::string& name = var.name;
					igPushID_Str("graph tree");
					igPushID_Str(name.c_str());
					igCheckbox(name.c_str(), &var.enabled);
					igSameLine(0, 0);
					if (igTreeNode_Str("")) {
						igCheckbox("log", &var.log);
				
						const char* polyModes[] = {"point", "line", "fill"};
						igCombo_Str_arr("polyMode", &var.polyMode, polyModes, numberof(polyModes), -1);
						
						igSliderFloat("alpha", &var.alpha, 0.f, 1.f, "%.3f", ImGuiSliderFlags_None);

						igSliderFloat("scale", &var.scale, 1e-10, 1e+10, "%.16f", ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_NoRoundToFormat); 

						igSliderInt("spacing", &var.step, 1, (int)size.s[0], "%d", ImGuiSliderFlags_None);
						
						igTreePop();
					}
					igPopID();
					igPopID();
				};

				Plot::Graph::Variable all = graph->variables[0];
				all.name = "all";
				Plot::Graph::Variable prevAll = all;
				addVarControls(all);
		
				auto applyFields = [&](auto fields) {
					for (auto field : fields) {
						if (prevAll.*field != all.*field) {
							for (Plot::Graph::Variable& var : graph->variables) {
								var.*field = all.*field;
							}
						}
					}
				};

				applyFields(std::vector<bool Plot::Graph::Variable::*>{
					&Plot::Graph::Variable::enabled,
					&Plot::Graph::Variable::log,
				});
				applyFields(std::vector<int Plot::Graph::Variable::*>{
					&Plot::Graph::Variable::polyMode,
					&Plot::Graph::Variable::step,
				});
				applyFields(std::vector<float Plot::Graph::Variable::*>{
					&Plot::Graph::Variable::alpha,
					&Plot::Graph::Variable::scale,
				});
				
				for (Plot::Graph::Variable& var : graph->variables) {
					addVarControls(var);
				}
			}
		}

		if (igCollapsingHeader_TreeNodeFlags("vector field", ImGuiTreeNodeFlags_None)) {
			igPushID_Str("vector field");
			std::vector<const char*> vectorFieldVarsCStrs = getCStrsFromStrVector(solver->getEquation()->vectorFieldVars);
			igCombo_Str_arr("variable", &vectorField->variable, vectorFieldVarsCStrs.data(), vectorFieldVarsCStrs.size(), -1); 
			
			igCheckbox("show", &showVectorField); // velocity field on/off
			//TODO velocity field variable
			igSliderFloat("scale", &vectorField->scale, 1e-10, 1e+10, "%.16f", ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_NoRoundToFormat); // velocity field size
			igPopID();
		}

		if (igCollapsingHeader_TreeNodeFlags("controls", ImGuiTreeNodeFlags_None)) {

			if (dim > 1) {
				bool isOrtho = view == viewOrtho;
				if (igButton(isOrtho ? "frustum" : "ortho", ImVec2())) {	// switching between ortho and frustum views
					if (!isOrtho) {
						view = viewOrtho;
					} else {
						view = viewFrustum;
					}
				}
			} else {
				view = viewOrtho;
			}

			if (igButton(doUpdate ? "pause" : "start", ImVec2())) doUpdate = !doUpdate; // start/stop simulation
			if (igButton("step", ImVec2())) doUpdate = 2;
			if (igButton("reset", ImVec2())) solver->resetState();

			// used fixed dt vs cfl #
			//TODO fixed dt means writing to the dtBuffer
			igInputFloat("cfl", &cfl, .1, 1, "%.3f", ImGuiInputTextFlags_None);

			// take screenshot
			if (igButton("screenshot", ImVec2())) {
				plot->screenshot();
			}

			// continuous screenshots
			if (igButton(createAnimation ? "stop frame dump" : "start frame dump", ImVec2())) {
				createAnimation = !createAnimation;
			}

			if (igButton("save", ImVec2())) {	// dump state of simulation
				solver->save();
			}
		}

#if 0
		{
			//TODO redirect stdout to here ...
			//or just override print() and io.write() ... anything else?
			//static char output[16384] = {'\0'};
			//igInputTextMultiline("##output", output, sizeof(output), ImVec2(), ImGuiInputTextFlags_ReadOnly);

			static char scriptInputBuffer[512] = {'\0'};
			if (igInputTextMultiline("##scriptInputBuffer", scriptInputBuffer, sizeof(scriptInputBuffer), ImVec2(),
				ImGuiInputTextFlags_AllowTabInput |
				ImGuiInputTextFlags_EnterReturnsTrue |
				ImGuiInputTextFlags_CtrlEnterForNewLine)
			) {
				//for redirection, check out freopen on stdout/stderr
				std::cout << "script input buffer was " << scriptInputBuffer << std::endl;
				try {
					lua.loadString(scriptInputBuffer);
				} catch (const Common::Exception& e) {
					std::cout << "failed with exception " << e.what() << std::endl;
				}
				//looks like setting the 'enter returns true' flag makes buffer clearing work as well
				memset(scriptInputBuffer, 0, sizeof(scriptInputBuffer));
			}
		}
#endif

		/*
		TODO

		slope limiter
		integrator
		fixed dt
		cfl #
		equation constants:
			NR:
				adm_BonaMasso_f
				adm_BonaMasso_df_dalpha
				speedOfLight
			Euler & MHD:
				gamma
			MHD:
				vaccuumPermeability
			Maxwell:
				permeability
				permittivity
				conductivity

		*/
	});

PROFILE_END_FRAME();
}

void HydroGPUApp::onSDLEvent(SDL_Event& event) {
	if (gui) gui->onSDLEvent(event);
	bool canHandleMouse = !igGetIO()->WantCaptureMouse;
	bool canHandleKeyboard = !igGetIO()->WantCaptureKeyboard;

	bool shiftDown = leftShiftDown || rightShiftDown;

	//Super::onSDLEvent only consists of ViewBehavior::onSDLEvent
	if (canHandleMouse) {
		Super::onSDLEvent(event);
	}

	switch (event.type) {
	case SDL_KEYUP:
		if (canHandleKeyboard) {
			if (event.key.keysym.sym == SDLK_s) {
				if (shiftDown) {
					solver->save();
				} else {
					plot->screenshot();
				}
			} else if (event.key.keysym.sym == SDLK_f) {
				if (shiftDown) {
					if (heatMap) heatMap->scale *= .5;
					if (iso3D) iso3D->scale *= .5;
				} else {
					if (heatMap) heatMap->scale *= 2.;
					if (iso3D) iso3D->scale *= 2.;
				}
				if (heatMap) std::cout << "heatMap->scale " << heatMap->scale << std::endl;
				if (iso3D) std::cout << "iso3D->scale " << iso3D->scale << std::endl;
			} else if (event.key.keysym.sym == SDLK_b) {
				if (graph) {
					for (Plot::Graph::Variable& var : graph->variables) {
						if (shiftDown) {
							var.scale *= .5;
						} else {
							var.scale *= 2.;
						}
						std::cout << "var " << var.name << " scale " << var.scale << std::endl;
					}
				}
			} else if (event.key.keysym.sym == SDLK_d) {
				if (shiftDown) {
					if (heatMap) heatMap->variable = (heatMap->variable + solver->getEquation()->displayVariables.size() - 1) % solver->getEquation()->displayVariables.size();
					if (iso3D) iso3D->variable = (iso3D->variable + solver->getEquation()->displayVariables.size() - 1) % solver->getEquation()->displayVariables.size();
				} else {
					if (heatMap) heatMap->variable = (heatMap->variable + 1) % solver->getEquation()->displayVariables.size();
					if (iso3D) iso3D->variable = (iso3D->variable + 1) % solver->getEquation()->displayVariables.size();
				}
				if (heatMap) std::cout << "heatMap->variable " << solver->getEquation()->displayVariables[heatMap->variable] << std::endl;
				if (iso3D) std::cout << "iso3D->variable " << solver->getEquation()->displayVariables[iso3D->variable] << std::endl;
			} else if (event.key.keysym.sym == SDLK_SPACE) {
				if (doUpdate) {
					std::cout << "stopping..." << std::endl;
					doUpdate = 0;
				} else {
					std::cout << "starting..." << std::endl;
					doUpdate = 1;
				}
			} else if (event.key.keysym.sym == SDLK_u) {
				std::cout << "step..." << std::endl;
				doUpdate = 2;
			} else if (event.key.keysym.sym == SDLK_r) {
				solver->resetState();
			} else if (event.key.keysym.sym == SDLK_t) {
				showTimestep = !showTimestep;
			} else if (event.key.keysym.sym == SDLK_v) {
				showVectorField = !showVectorField;
				std::cout << "vector field " << (showVectorField ? "enabled" : "disabled") << std::endl;
			} else if (event.key.keysym.sym == SDLK_c) {
				if (shiftDown) {
					vectorField->scale *= .5f;
				} else {
					vectorField->scale *= 2.f;
				}
				std::cout << "vector field scale " << vectorField->scale << std::endl;
			}
		}
		break;
	}
}

}

GLAPP_MAIN(HydroGPU::HydroGPUApp)
