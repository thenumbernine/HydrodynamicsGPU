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

#include "HydroGPU/Plot/CameraOrtho.h"
#include "HydroGPU/Plot/CameraFrustum.h"
#include "HydroGPU/Plot/Graph.h"
#include "HydroGPU/Plot/HeatMap.h"
#include "HydroGPU/Plot/Iso3D.h"
#include "HydroGPU/Plot/Plot.h"
#include "HydroGPU/Plot/VectorField.h"

#include "HydroGPU/HydroGPUApp.h"

#include "ImGuiCommon/ImGuiCommon.h"
#include "Profiler/Profiler.h"
#include "Common/Exception.h"
#include "Common/File.h"
#include "Common/Macros.h"

#include <SDL2/SDL.h>
#include <OpenCL/cl.hpp>
#include <OpenGL/gl.h>
#include <OpenGL/OpenGL.h>
#include <iostream>


static std::string makeComboStr(const std::vector<std::string>& v) {
	std::string result;
	for (const std::string& s : v) {
		result.append(s).append(1, '\0');
	}
	return result;
}

namespace HydroGPU {

HydroGPUApp::HydroGPUApp()
: Super()
, gradientTex(GLuint())
, configFilename("config.lua")
, solverName("EulerBurgers")
, dim(0)
, doUpdate(1)
, maxFrames(-1)
, currentFrame(-1)
, useFixedDT(false)
, fixedDT(.001f)
, cfl(.5f)
, showHeatMap(true)
, showIso3D(true)
, useGravity(false)
, gaussSeidelMaxIter(20)
, showVectorField(true)
, createAnimation(false)
, showTimestep(false)
, leftButtonDown(false)
, rightButtonDown(false)
, leftShiftDown(false)
, rightShiftDown(false)
, leftGuiDown(false)
, rightGuiDown(false)
, aspectRatio(0.f)
{
#define MAKE_SOLVER(solver) std::pair<std::string, std::function<std::shared_ptr<Solver::Solver>()>>(#solver, [=]()->std::shared_ptr<Solver::Solver>{ return std::make_shared<Solver::solver>(this); })
	solverGensForEqns = {
		{"Euler", {
			MAKE_SOLVER(EulerBurgers),
			MAKE_SOLVER(EulerHLL),
			MAKE_SOLVER(EulerHLLC),
			MAKE_SOLVER(EulerRoe),
		}},
		{"SRHD", {
			MAKE_SOLVER(SRHDRoe),
		}},
		{"MHD", {
			MAKE_SOLVER(MHDBurgers),
			MAKE_SOLVER(MHDHLLC),
			MAKE_SOLVER(MHDRoe),
		}},
		{"Maxwell", {
			MAKE_SOLVER(MaxwellRoe),
		}},
		{"ADM1D", {
			MAKE_SOLVER(ADM1DRoe),
		}},
		{"ADM3D", {
			MAKE_SOLVER(ADM3DRoe),
		}},
		{"BSSNOK", {
			MAKE_SOLVER(BSSNOKRoe),
		}},
	};
#undef MAKE_SOLVER

	for (const std::pair<std::string, std::vector<std::pair<std::string, std::function<std::shared_ptr<Solver::Solver>()>>>>& p : solverGensForEqns) {
		const std::string& equationName = p.first;
		equationNames.push_back(equationName);
		std::vector<std::string> solverNamesForEqn;
		for (const std::pair<std::string, std::function<std::shared_ptr<Solver::Solver>()>>& q : p.second) {
			const std::string& thisSolverName = q.first;
			solverNamesForEqn.push_back(thisSolverName);
			solverGens.push_back(q);
		}
		solverNamesSeparatedForEqn[equationName] = makeComboStr(solverNamesForEqn);
	}
	equationNamesSeparated = makeComboStr(equationNames);
	equationIndex = 0;

	for (int i = 0; i < 4; ++i) {
		size.s[i] = 1;	//default each dimension to a point.  so if lua doesn't define it then the dimension will be ignored
		xmin.s[i] = -.5f;
		xmax.s[i] = .5f;
	}
}

int HydroGPUApp::main(const std::vector<std::string>& args) {
	for (int i = 1; i < args.size(); ++i) {
		if (i < args.size()-1 && args[i] == "-e") {
			configString = args[++i];
		} else {
			configFilename = args[++i];
		}
	}
	return Super::main(args);
}

void HydroGPUApp::init() {

	{
		std::string extensionStr = (char*)glGetString(GL_EXTENSIONS);
		std::istringstream iss(extensionStr);
		std::vector<std::string> extensions;
		std::copy(std::istream_iterator<std::string>(iss), std::istream_iterator<std::string>(), std::back_inserter<std::vector<std::string>>(extensions));
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

	for (const std::string& eqnName : equationNames) {
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
		initCondNamesSeparatedForEqns[eqnName] = makeComboStr(initCondNames);
	}

	lua["maxFrames"] >> maxFrames;
	lua["showTimestep"] >> showTimestep;
	lua["useFixedDT"] >> useFixedDT;
	lua["fixedDT"] >> fixedDT;
	lua["cfl"] >> cfl;
	lua["useGravity"] >> useGravity;
	lua["gaussSeidelMaxIter"] >> gaussSeidelMaxIter;

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

	Super::init();

	gui = std::make_shared<ImGuiCommon::ImGuiCommon>(window);

	clCommon = std::make_shared<CLCommon::CLCommon>(
		useGPU,
		/*verbose=*/true,
		/*pickDevice=*/[&](const std::vector<cl::Device>& devices) -> std::vector<cl::Device>::const_iterator {
			return std::find_if(devices.begin(), devices.end(), [&](const cl::Device& device) -> bool {
				std::vector<std::string> extensions = CLCommon::getExtensions(device);
				return (std::find(extensions.begin(), extensions.end(), "cl_khr_gl_sharing") != extensions.end()
					|| std::find(extensions.begin(), extensions.end(), "cl_APPLE_gl_sharing") != extensions.end())
					&& std::find(extensions.begin(), extensions.end(), "cl_khr_fp64") != extensions.end();
			});
		});

	glEnable(GL_DEPTH_TEST);

	//gradient texture
	{
		glGenTextures(1, &gradientTex);
		glBindTexture(GL_TEXTURE_1D, gradientTex);

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
			if (ci >= numberof(colors)) {
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
		glTexImage1D(GL_TEXTURE_1D, 0, GL_RGB, width, 0, GL_RGB, GL_UNSIGNED_BYTE, data);
		
		//glGenerateMipmap(GL_TEXTURE_1D);
		//glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);	//isobars look good with mipmapping
		
		glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);	//heatmaps do not
		
		glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glBindTexture(GL_TEXTURE_1D, 0);
	}

	gradientTexMem = cl::ImageGL(clCommon->context, CL_MEM_READ_ONLY, GL_TEXTURE_1D, 0, gradientTex);

	lua["solverName"] >> solverName;
	std::cout << "solverName " << solverName << std::endl;
	{
		std::vector<std::pair<std::string, std::function<std::shared_ptr<Solver::Solver>()>>>::iterator i
			= find_if(solverGens.begin(), solverGens.end(),
				[&](const std::pair<std::string, std::function<std::shared_ptr<Solver::Solver>()>>& p)->bool{
					return p.first == solverName;
				});
		if (i == solverGens.end()) {
			throw Common::Exception() << "unknown solver " << solverName;
		}
		solver = i->second();
		solver->init();	//...now that the vtable is in place
		solver->resetState();
	}

	//find the index of the equation associated with the selected solver (for the combo box)
	for (equationIndex = 0; equationIndex < equationNames.size(); ++equationIndex) {
		if (solver->equation->name() == equationNames[equationIndex]) break;
	}
	if (equationIndex == equationNames.size()) equationIndex = 0;

	//find the solver in the equations
	for (solverForEqnIndex = 0; solverForEqnIndex < solverGensForEqns[equationIndex].second.size(); ++solverForEqnIndex) {
		if (solver->name() == solverGensForEqns[equationIndex].second[solverForEqnIndex].first) break;
	}
	if (solverForEqnIndex == solverGensForEqns[equationIndex].second.size()) solverForEqnIndex = 0;

	{
		std::string initCondName;
		if ((lua["initCondName"] >> initCondName).good()) {
			std::string equationName = equationNames[equationIndex];
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
				solver->equation->vectorFieldVars.begin(),
				solver->equation->vectorFieldVars.end(),
				variableName)
				- solver->equation->vectorFieldVars.begin();
			if (vectorFieldVariable == solver->equation->vectorFieldVars.size()) {
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

	camera = cameraOrtho = std::make_shared<Plot::CameraOrtho>(this);
	cameraFrustum = std::make_shared<Plot::CameraFrustum>(this);
	{
		std::string mode;
		lua["camera"]["mode"] >> mode;
		if (mode == "ortho") {
			camera = cameraOrtho;
		} else if (mode == "frustum") {
			camera = cameraFrustum;
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
				//TODO LuaCxx iterator
				for (int i = 0; i < graphVariablesRef.len(); ++i) {
					std::string varName;
					if ((graphVariablesRef[i+1] >> varName).good()) {
						graphVariableNames.push_back(varName);
					}
				}
			}

			if (graphVariableNames.empty()) {
				graphVariableNames = solver->equation->displayVariables;
			}

			//make a mapping from names to indexes
			const std::vector<std::string>& displayVariables = solver->equation->displayVariables;
			for (std::vector<std::string>::const_iterator i = displayVariables.begin(); i != displayVariables.end(); ++i) {
				int displayVarIndex = i - displayVariables.begin();
				if (displayVarIndex >= 0 && displayVarIndex < displayVariables.size()) {
					graph->variables[displayVarIndex].enabled = true;
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
					solver->equation->displayVariables.begin(),
					solver->equation->displayVariables.end(),
					variableName)
					- solver->equation->displayVariables.begin();
				if (heatMap->variable == solver->equation->displayVariables.size()) {
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
					solver->equation->displayVariables.begin(),
					solver->equation->displayVariables.end(),
					variableName)
					- solver->equation->displayVariables.begin();
				if (iso3D->variable == solver->equation->displayVariables.size()) {
					iso3D->variable = 0;
					std::cerr << "couldn't interpret display method " << variableName << std::endl;
				}
			}		
		}
	}

	for (int i = 0; i < 3; ++i) {
		for (int minmax = 0; minmax < 2; ++minmax) {
			if (boundaryMethodNames[i][minmax].empty()) continue;
			boundaryMethods(i,minmax) = std::find(
				solver->equation->boundaryMethods.begin(),
				solver->equation->boundaryMethods.end(),
				boundaryMethodNames[i][minmax])
				- solver->equation->boundaryMethods.begin();
			if (boundaryMethods(i,minmax) == solver->equation->boundaryMethods.size()) {
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

void HydroGPUApp::shutdown() {
	gui = nullptr;	//dealloc and shutdown before sdl shuts down
	glDeleteTextures(1, &gradientTex);
	clCommon.reset();
	Super::shutdown();
}

void HydroGPUApp::resize(int width, int height) {
	Super::resize(width, height);	//viewport
	screenSize = Tensor::Vector<int,2>(width, height);
	aspectRatio = (float)screenSize(0) / (float)screenSize(1);
}

void HydroGPUApp::update() {
PROFILE_BEGIN_FRAME()

	++currentFrame;
	if (currentFrame == maxFrames) {
		doUpdate = 0;
	}

	Super::update();	//glclear

	bool guiDown = leftGuiDown || rightGuiDown;
	if (rightButtonDown || (leftButtonDown && guiDown)) {
		//TODO - direct edit of the field ... solver->addDrop();
	}

	if (doUpdate) {
		solver->update();
		if (doUpdate == 2) doUpdate = 0;
	}

	camera->setupProjection();
	camera->setupModelview();

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
	gui->update([&](){
		//how do you change the window title from "Debug"?

		if (ImGui::CollapsingHeader("setup")) {
			//list equations
			int lastEquationIndex = equationIndex;
			ImGui::Combo("equation", &equationIndex, equationNamesSeparated.c_str());
			if (lastEquationIndex != equationIndex) {

				//if we are changing equations then (most likely) our initial conditions will be invalid
				//(except cases where they belong to the same groups, I think, unless the resetState conversion works diferently,
				// either way, to be safe...)
				//we will have to reset the initial condition to something that certainly belongs to this equation
				initCondIndex = 0;
				std::string initCondName = initCondNamesForEqns[equationNames[equationIndex]][initCondIndex];
				lua["initConds"][initCondName]["setup"]();

				solverForEqnIndex = 0;
				std::cout << "setting equation to " << solverGensForEqns[equationIndex].first << std::endl;
				solver = solverGensForEqns[equationIndex].second[solverForEqnIndex].second();
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
			ImGui::Combo("solver", &solverForEqnIndex, solverNamesSeparatedForEqn[equationNames[equationIndex]].c_str());
			if (lastSolverForEqnIndex != solverForEqnIndex) {
				std::cout << "setting solver to " << solverGensForEqns[equationIndex].second[solverForEqnIndex].first << std::endl;
				std::shared_ptr<Solver::Solver> newSolver = solverGensForEqns[equationIndex].second[solverForEqnIndex].second();
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
			ImGui::Combo("init.cond.", &initCondIndex, initCondNamesSeparatedForEqns[equationNames[equationIndex]].c_str());
			if (lastInitCondIndex != initCondIndex) {
				std::string initCondName = initCondNamesForEqns[equationNames[equationIndex]][initCondIndex];
				std::cout << "setting up initial condition " << initCondName << std::endl;
				lua["initConds"][initCondName]["setup"]();
			}
		}

		if (ImGui::CollapsingHeader("boundaries")) {
			std::vector<std::string> methods = solver->equation->boundaryMethods;
			methods.insert(methods.begin(), "NONE");
			std::string s = makeComboStr(methods);
			for (int i = 0; i < dim; ++i) {
				for (int minmax = 0; minmax < 2; ++minmax) {
					std::stringstream comboTitle;
					comboTitle << (char)('x'+i) << " " << (minmax ? "max" : "min");
					++boundaryMethods(i,minmax);
					ImGui::Combo(comboTitle.str().c_str(), &boundaryMethods(i,minmax), s.c_str());
					--boundaryMethods(i,minmax);
				}
			}
		}

		if (heatMap) {
			if (ImGui::CollapsingHeader("heat map")) {
				ImGui::Checkbox("show heatMap", &showHeatMap);

				std::string s = makeComboStr(solver->equation->displayVariables);
				ImGui::Combo("heatMap variable ", &heatMap->variable, s.c_str());

				float logScale = log(heatMap->scale);
				ImGui::SliderFloat("heatMap scale ", &logScale, -10.f, 10.f);
				heatMap->scale = exp(logScale);
			
				ImGui::Checkbox("heatMap log scale", &heatMap->useLog);
			
				ImGui::SliderFloat("heatMap alpha", &heatMap->alpha, 0.f, 1.f);
			}
		}

		if (iso3D) {
			if (ImGui::CollapsingHeader("isosurfaces")) {
				ImGui::Checkbox("show isosurfaces ", &showIso3D);

				std::string s = makeComboStr(solver->equation->displayVariables);
				ImGui::Combo("isosurface variable", &iso3D->variable, s.c_str());

				float logScale = log(iso3D->scale);
				ImGui::SliderFloat("isosurface scale ", &logScale, -10.f, 10.f);
				iso3D->scale = exp(logScale);
			
				ImGui::Checkbox("isosurface log scale", &iso3D->useLog);
				
				ImGui::SliderFloat("isosurface alpha", &iso3D->alpha, 0.f, 1.f);
			}
		}

		if (graph) {
			if (ImGui::CollapsingHeader("graph")) {
				std::function<void(Plot::Graph::Variable&)> addVarControls = [&](Plot::Graph::Variable& var){
					const std::string& name = var.name;
					ImGui::Checkbox((name + std::string(" enabled")).c_str(), &var.enabled);
					ImGui::SameLine();
					if (ImGui::TreeNode((name + std::string(" tree")).c_str())) {
						
						ImGui::Checkbox((std::string("log ") + name).c_str(), &var.log);
					
						ImGui::Combo((std::string("polyMode ") + name).c_str(), &var.polyMode, makeComboStr({"point", "line", "fill"}).c_str());
						
						ImGui::SliderFloat((std::string("alpha ") + name).c_str(), &var.alpha, 0.f, 1.f);

						float logScale = log(var.scale) / log(10);
						ImGui::SliderFloat((std::string("scale ") + name).c_str(), &logScale, -10, 10);
						var.scale = pow(10, logScale);

						ImGui::SliderInt((std::string("spacing ") + name).c_str(), &var.step, 1, (int)size.s[0]);
						
						ImGui::TreePop();
					}
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

		if (ImGui::CollapsingHeader("vector field")) {
			std::string s = makeComboStr(solver->equation->vectorFieldVars);
			ImGui::Combo("vector field variable", &vectorField->variable, s.c_str());
			
			ImGui::Checkbox("show vectorField", &showVectorField); // velocity field on/off
			//TODO velocity field variable
			ImGui::Text("log scale");
			float logVectorFieldScale = log(vectorField->scale);
			ImGui::SliderFloat("v.f.s.", &logVectorFieldScale, -10.f, 10.f); // velocity field size
			vectorField->scale = exp(logVectorFieldScale);
		}

		if (ImGui::CollapsingHeader("controls")) {

			if (dim > 1) {
				bool isOrtho = camera == cameraOrtho;
				if (ImGui::Button(isOrtho ? "frustum" : "ortho")) {	// switching between ortho and frustum views
					if (!isOrtho) {
						camera = cameraOrtho;
					} else {
						camera = cameraFrustum;
					}
				}
			} else {
				camera = cameraOrtho;
			}

			if (ImGui::Button(doUpdate ? "pause" : "start")) doUpdate = !doUpdate; // start/stop simulation
			if (ImGui::Button("step")) doUpdate = 2;
			if (ImGui::Button("reset")) solver->resetState();

			// used fixed dt vs cfl #
			//TODO fixed dt means writing to the dtBuffer
			ImGui::InputFloat("cfl", &cfl);

			// take screenshot
			if (ImGui::Button("screenshot")) {
				plot->screenshot();
			}

			// continuous screenshots
			if (ImGui::Button(createAnimation ? "stop frame dump" : "start frame dump")) {
				createAnimation = !createAnimation;
			}

			if (ImGui::Button("save")) {	// dump state of simulation
				solver->save();
			}
		}

#if 0
		{
			//TODO redirect stdout to here ...
			//or just override print() and io.write() ... anything else?
			//static char output[16384] = {'\0'};
			//ImGui::InputTextMultiline("##output", output, sizeof(output), ImVec2(), ImGuiInputTextFlags_ReadOnly);

			static char scriptInputBuffer[512] = {'\0'};
			if (ImGui::InputTextMultiline("##scriptInputBuffer", scriptInputBuffer, sizeof(scriptInputBuffer), ImVec2(),
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

void HydroGPUApp::sdlEvent(SDL_Event& event) {
	gui->sdlEvent(event);
	bool canHandleMouse = !ImGui::GetIO().WantCaptureMouse;
	bool canHandleKeyboard = !ImGui::GetIO().WantCaptureKeyboard;

	bool shiftDown = leftShiftDown || rightShiftDown;
	bool guiDown = leftGuiDown || rightGuiDown;

	switch (event.type) {
	case SDL_MOUSEMOTION:
		if (canHandleMouse) {
			int dx = event.motion.xrel;
			int dy = event.motion.yrel;
			if (leftButtonDown && !guiDown) {
				if (shiftDown) {
					if (dy) {
						camera->mouseZoom(dy);
					}
				} else {
					if (dx || dy) {
						camera->mousePan(dx, dy);
					}
				}
			}
		}
		break;
	case SDL_MOUSEBUTTONDOWN:
		if (event.button.button == SDL_BUTTON_LEFT) {
			leftButtonDown = true;
		}
		if (event.button.button == SDL_BUTTON_RIGHT) {
			rightButtonDown = true;
		}
		break;
	case SDL_MOUSEBUTTONUP:
		if (event.button.button == SDL_BUTTON_LEFT) {
			leftButtonDown = false;
		}
		if (event.button.button == SDL_BUTTON_RIGHT) {
			rightButtonDown = false;
		}
		break;
	case SDL_KEYDOWN:
		if (event.key.keysym.sym == SDLK_LSHIFT) {
			leftShiftDown = true;
		} else if (event.key.keysym.sym == SDLK_RSHIFT) {
			rightShiftDown = true;
		} else if (event.key.keysym.sym == SDLK_LGUI) {
			leftGuiDown = true;
		} else if (event.key.keysym.sym == SDLK_RGUI) {
			rightGuiDown = true;
		}
		break;
	case SDL_KEYUP:
		if (event.key.keysym.sym == SDLK_LSHIFT) {
			leftShiftDown = false;
		} else if (event.key.keysym.sym == SDLK_RSHIFT) {
			rightShiftDown = false;
		} else if (event.key.keysym.sym == SDLK_LGUI) {
			leftGuiDown = false;
		} else if (event.key.keysym.sym == SDLK_RGUI) {
			rightGuiDown = false;
		} else if (canHandleKeyboard) {
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
					if (heatMap) heatMap->variable = (heatMap->variable + solver->equation->displayVariables.size() - 1) % solver->equation->displayVariables.size();
					if (iso3D) iso3D->variable = (iso3D->variable + solver->equation->displayVariables.size() - 1) % solver->equation->displayVariables.size();
				} else {
					if (heatMap) heatMap->variable = (heatMap->variable + 1) % solver->equation->displayVariables.size();
					if (iso3D) iso3D->variable = (iso3D->variable + 1) % solver->equation->displayVariables.size();
				}
				if (heatMap) std::cout << "heatMap->variable " << solver->equation->displayVariables[heatMap->variable] << std::endl;
				if (iso3D) std::cout << "iso3D->variable " << solver->equation->displayVariables[iso3D->variable] << std::endl;
			} else if (event.key.keysym.sym == SDLK_u) {
				if (doUpdate) {
					std::cout << "stopping..." << std::endl;
					doUpdate = 0;
				} else {
					if (shiftDown) {
						std::cout << "step..." << std::endl;
						doUpdate = 2;
					} else {
						std::cout << "starting..." << std::endl;
						doUpdate = 1;
					}
				}
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
