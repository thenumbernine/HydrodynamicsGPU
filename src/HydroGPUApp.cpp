#include "HydroGPU/HydroGPUApp.h"
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
#include "HydroGPU/Plot/VectorField.h"
#include "HydroGPU/Plot/Plot1D.h"
#include "HydroGPU/Plot/Plot2D.h"
#include "HydroGPU/Plot/Plot3D.h"
#include "HydroGPU/Plot/CameraOrtho.h"
#include "HydroGPU/Plot/CameraFrustum.h"
#include "HydroGPU/Plot/Graph.h"
#include "Profiler/Profiler.h"
#include "Common/Exception.h"
#include "Common/File.h"
#include "Common/Macros.h"
#include <SDL2/SDL.h>
#include <OpenCL/cl.hpp>
#include <OpenGL/gl.h>
#include <OpenGL/OpenGL.h>
#include <iostream>

namespace HydroGPU {

//have to keep these updated with HydroGPU/Shared/Common.h

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
, useHeatMap(true)
, heatMapVariable(0)
, heatMapColorScale(2.f)
, useGravity(false)
, gaussSeidelMaxIter(20)
, showVectorField(true)
, vectorFieldScale(.125f)
, leftButtonDown(false)
, rightButtonDown(false)
, leftShiftDown(false)
, rightShiftDown(false)
, leftGuiDown(false)
, rightGuiDown(false)
, showTimestep(false)
{
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
	std::cout << "loading config file " << configFilename << std::endl;
	lua.loadFile(configFilename);
	if (!configString.empty()) {
		std::cout << "loading config string " << configString << std::endl;
		lua.loadString(configString);
	}

	bool useGPU = true;
	lua.ref()["useGPU"] >> useGPU;
	for (int i = 0; i < 3; ++i) {
		if (!lua.ref()["size"].isNil()) lua.ref()["size"][i+1] >> size.s[i];
		if (!lua.ref()["xmin"].isNil()) lua.ref()["xmin"][i+1] >> xmin.s[i];
		if (!lua.ref()["xmax"].isNil()) lua.ref()["xmax"][i+1] >> xmax.s[i];
	}
	
	std::vector<std::vector<std::string>> boundaryMethodNames(3);
	for (int i = 0; i < 3; ++i) {
		boundaryMethodNames[i].resize(2);
		if (!lua.ref()["boundaryMethods"].isNil()) {
			if (lua.ref()["boundaryMethods"][i+1].isTable()) {
				lua.ref()["boundaryMethods"][i+1]["min"] >> boundaryMethodNames[i][0];
				lua.ref()["boundaryMethods"][i+1]["max"] >> boundaryMethodNames[i][1];
			}
		}
	}
	
	lua.ref()["maxFrames"] >> maxFrames;
	lua.ref()["showTimestep"] >> showTimestep;
	lua.ref()["solverName"] >> solverName;
	lua.ref()["useFixedDT"] >> useFixedDT;
	lua.ref()["fixedDT"] >> fixedDT;
	lua.ref()["cfl"] >> cfl;
	lua.ref()["heatMapColorScale"] >> heatMapColorScale;
	lua.ref()["useGravity"] >> useGravity;
	lua.ref()["gaussSeidelMaxIter"] >> gaussSeidelMaxIter;

	std::string heatMapVariableName;
	lua.ref()["heatMapVariable"] >> heatMapVariableName;

	lua.ref()["showVectorField"] >> showVectorField;
	lua.ref()["vectorFieldScale"] >> vectorFieldScale;

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

		float colors[][3] = {
			{0, 0, 0},
			{0, 0, 1},
			{0, 1, 1},
			{1, 1, 0},
			{1, 0, 0}
		};

		const int width = 256;
		unsigned char data[width*3];
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

		glTexImage1D(GL_TEXTURE_1D, 0, GL_RGB, width, 0, GL_RGB, GL_UNSIGNED_BYTE, data);
		glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glBindTexture(GL_TEXTURE_1D, 0);
	}

	gradientTexMem = cl::ImageGL(clCommon->context, CL_MEM_READ_ONLY, GL_TEXTURE_1D, 0, gradientTex);

	//construct the solver
	std::cout << "solverName " << solverName << std::endl;
	if (solverName == "EulerBurgers") {
		solver = std::make_shared<HydroGPU::Solver::EulerBurgers>(this);
	} else if (solverName == "EulerHLL") {
		solver = std::make_shared<HydroGPU::Solver::EulerHLL>(this);
	} else if (solverName == "EulerHLLC") {
		solver = std::make_shared<HydroGPU::Solver::EulerHLLC>(this);
	} else if (solverName == "EulerRoe") {
		solver = std::make_shared<HydroGPU::Solver::EulerRoe>(this);
	} else if (solverName == "SRHDRoe") {
		solver = std::make_shared<HydroGPU::Solver::SRHDRoe>(this);
	} else if (solverName == "MHDBurgers") {
		solver = std::make_shared<HydroGPU::Solver::MHDBurgers>(this);
	} else if (solverName == "MHDHLLC") {
		solver = std::make_shared<HydroGPU::Solver::MHDHLLC>(this);
	} else if (solverName == "MHDRoe") {
		solver = std::make_shared<HydroGPU::Solver::MHDRoe>(this);
	} else if (solverName == "MaxwellRoe") {
		solver = std::make_shared<HydroGPU::Solver::MaxwellRoe>(this);
	} else if (solverName == "ADM1DRoe") {
		solver = std::make_shared<HydroGPU::Solver::ADM1DRoe>(this);
	} else if (solverName == "ADM3DRoe") {
		solver = std::make_shared<HydroGPU::Solver::ADM3DRoe>(this);
	} else if (solverName == "BSSNOKRoe") {
		solver = std::make_shared<HydroGPU::Solver::BSSNOKRoe>(this);
	} else {
		throw Common::Exception() << "unknown solver " << solverName;
	}
	
	solver->init();	//..now that the vtable is in place

	//needs solver->program to be created
	vectorField = std::make_shared<HydroGPU::Plot::VectorField>(solver);

	if (lua.ref()["camera"].isNil()) {
		throw Common::Exception() << "unknown camera";
	}
	
	{
		std::string mode;
		lua.ref()["camera"]["mode"] >> mode;
		if (mode == "ortho") {
			camera = std::make_shared<HydroGPU::Plot::CameraOrtho>(this);
		} else if (mode == "frustum") {
			camera = std::make_shared<HydroGPU::Plot::CameraFrustum>(this);
		} else {
			throw Common::Exception() << "unknown camera mode";
		}
	}
	
	//needs solver->program to be created
	switch(dim) {
	case 1:
		plot = std::make_shared<HydroGPU::Plot::Plot1D>(this);
		break;
	case 2:
		plot = std::make_shared<HydroGPU::Plot::Plot2D>(this);
		break;
	case 3:
		plot = std::make_shared<HydroGPU::Plot::Plot3D>(this);
		break;
	}
	plot->init();

	{
		bool useGraph = false;
		if ((lua.ref()["useGraph"] >> useGraph).good() && useGraph) {
		
			std::vector<std::string> graphVariableNames;
			LuaCxx::Ref graphVariablesRef = lua.ref()["graphVariables"];
			if (graphVariablesRef.isTable()) {
				//TODO LuaCxx iterator
				for (int i = 0; i < graphVariablesRef.len(); ++i) {
					std::string varName;
					if ((graphVariablesRef[i+1] >> varName).good()) {
						graphVariableNames.push_back(varName);
					}
				}
			}
			
			//make a mapping from names to indexes
			std::map<std::string, int> varIndexForName;
			const std::vector<std::string>& displayVariables = solver->equation->displayVariables;
			for (std::vector<std::string>::const_iterator i = displayVariables.begin(); i != displayVariables.end(); ++i) {
				varIndexForName[*i] = i - displayVariables.begin();
			}

			//if we weren't given any then assign it all vars in the equation 
			if (graphVariableNames.empty()) {
				graphVariableNames = displayVariables;
			}
			
			//now verify that the variables are legit, complain otherwise
			std::vector<int> graphVariables;
			for (const std::string& graphVarName : graphVariableNames) {
				std::map<std::string, int>::iterator loc = varIndexForName.find(graphVarName);
				if (loc == varIndexForName.end()) {
					throw Common::Exception() << "couldn't find graph variable " << graphVarName << " among display variables";
				}
				graphVariables.push_back(loc->second);
			}

			graph = std::make_shared<HydroGPU::Plot::Graph>(this);
			graph->variables = graphVariables;

			lua.ref()["graphScale"] >> graph->scale;

			{
				int i = 0;
				LuaCxx::Ref graphStepRef = lua.ref()["graphStep"];
				for (; i < dim; ++i) {
					graph->step(i) = size.s[i] >> 5;
					if (!graphStepRef.isTable()) {
						continue;
					}
					if (i >= graphStepRef.len()) {
						continue;
					}
					if (!(graphStepRef[i+1] >> graph->step(i)).good()) {
						continue;
					}
				}
				for (int i = 0; i < 3; ++i) {
					if (graph->step(i) == 0) graph->step(i) = 1;
				}
			}
		}
	}

	solver->resetState();

	lua.ref()["useHeatMap"] >> useHeatMap;

	heatMapVariable = std::find(
		solver->equation->displayVariables.begin(), 
		solver->equation->displayVariables.end(),
		heatMapVariableName) 
		- solver->equation->displayVariables.begin();
	if (heatMapVariable == solver->equation->displayVariables.size()) {
		throw Common::Exception() << "couldn't interpret display method " << heatMapVariableName;
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
	glDeleteTextures(1, &gradientTex);
	clCommon.reset();
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
	solver->app->camera->setupModelview();
	if (useHeatMap) plot->display();	
	vectorField->display();
	if (graph) graph->display();
PROFILE_END_FRAME();
}

void HydroGPUApp::sdlEvent(SDL_Event& event) {
	bool shiftDown = leftShiftDown || rightShiftDown;
	bool guiDown = leftGuiDown || rightGuiDown;

	switch (event.type) {
	case SDL_MOUSEMOTION:
		{
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
		} else if (event.key.keysym.sym == SDLK_s) {
			if (shiftDown) {
				solver->save();
			} else {
				solver->screenshot();
			}
		} else if (event.key.keysym.sym == SDLK_f) {
			if (shiftDown) {
				heatMapColorScale *= .5;
			} else {
				heatMapColorScale *= 2.;
			}
			std::cout << "heatMapColorScale " << heatMapColorScale << std::endl;
		} else if (event.key.keysym.sym == SDLK_b) {
			if (graph) {
				if (shiftDown) {
					graph->scale *= .5;
				} else {
					graph->scale *= 2.;
				}
				std::cout << "graph->scale " << graph->scale << std::endl;
			} else {
				//TODO always instanciate a graph, add controls for picking variables
				//TODO that gui ...
				std::cout << "no graph present" << std::endl;
			}
		} else if (event.key.keysym.sym == SDLK_d) {
			if (shiftDown) {
				heatMapVariable = (heatMapVariable + solver->equation->displayVariables.size() - 1) % solver->equation->displayVariables.size();
			} else {
				heatMapVariable = (heatMapVariable + 1) % solver->equation->displayVariables.size();
			}
			std::cout << "display " << solver->equation->displayVariables[heatMapVariable] << std::endl;
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
			std::cout << "velocity field " << (showVectorField ? "enabled" : "disabled") << std::endl;
		} else if (event.key.keysym.sym == SDLK_c) {
			if (shiftDown) {
				vectorFieldScale *= .5f;
			} else {
				vectorFieldScale *= 2.f;
			}
		}
		break;
	}
}

}

GLAPP_MAIN(HydroGPU::HydroGPUApp)

