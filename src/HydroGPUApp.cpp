#include "HydroGPU/HydroGPUApp.h"
#include "HydroGPU/RoeSolver2D.h"
#include "HydroGPU/BurgersSolver2D.h"
#include "Profiler/Profiler.h"
#include "Common/Exception.h"
#include "Common/File.h"
#include "Common/Macros.h"
#include <SDL2/SDL.h>
#include <OpenGL/gl.h>
#include <OpenGL/OpenGL.h>
#include <iostream>

//helper functions
namespace LuaCxx {
template<> void fromC<real2> (lua_State *L, const real2& value) { 
	lua_newtable(L);
	int t = lua_gettop(L);
	for (int i = 0; i < 2; ++i) {
		lua_pushnumber(L, value.s[i]);
		lua_rawseti(L, t, i+1);
	}
}

template<> real4 toC<real4>(lua_State *L, int loc) {
	real4 result; 
	for (int i = 0; i < 4; ++i) { 
		lua_rawgeti(L, loc, i+1); 
		result.s[i] = lua_tonumber(L, -1);
		lua_pop(L,1);
	}
	return result;
}
}

//have to keep these updated with HydroGPU/Shared/Common.h

std::vector<std::string> displayMethodNames = std::vector<std::string>{
	"density",
	"velocity",
	"pressure",
	"gravity potential",
};

std::vector<std::string> boundaryMethodNames = std::vector<std::string>{
	"periodic",
	"mirror",
	"freeflow",
};

HydroGPUApp::HydroGPUApp()
: Super()
, gradientTex(GLuint())
, configFilename("config.lua")
, solverName("Burgers")
, dim(2)
, doUpdate(1)
, maxFrames(-1)
, currentFrame(0)
, useFixedDT(false)
, fixedDT(.001f)
, cfl(.5f)
, displayMethod(DISPLAY_DENSITY)
, displayScale(2.f)
, boundaryMethods(BOUNDARY_PERIODIC, BOUNDARY_PERIODIC, BOUNDARY_PERIODIC)
, useGravity(true)
, gamma(1.4)
, leftButtonDown(false)
, rightButtonDown(false)
, leftShiftDown(false)
, rightShiftDown(false)
, leftGuiDown(false)
, rightGuiDown(false)
{
	for (int i = 0; i < 3; ++i) {
		size.s[i] = 512;
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
	//config before Super::init so we can provide it 'useGPU'
	std::cout << "loading config file " << configFilename << std::endl;
	lua.loadFile(configFilename);
	if (!configString.empty()) {
		std::cout << "loading config string " << configString << std::endl;
		lua.loadString(configString);
	}
	
	lua["useGPU"] >> useGPU;
	lua["dim"] >> dim;
	for (int i = 0; i < 3; ++i) {
		if (!lua["size"].isNil()) lua["size"][i+1] >> size.s[i];
		if (!lua["xmin"].isNil()) lua["xmin"][i+1] >> xmin.s[i];
		if (!lua["xmax"].isNil()) lua["xmax"][i+1] >> xmax.s[i];
		if (!lua["boundaryMethods"].isNil()) {
			std::string boundaryMethodName;
			if ((lua["boundaryMethods"][i+1] >> boundaryMethodName).good()) {
				boundaryMethods(i) = std::find(boundaryMethodNames.begin(), boundaryMethodNames.end(), boundaryMethodName) - boundaryMethodNames.begin();
			}
		}
	}
	lua["maxFrames"] >> maxFrames;
	lua["solverName"] >> solverName;
	lua["useFixedDT"] >> useFixedDT;
	lua["cfl"] >> cfl;
	lua["gamma"] >> gamma;
	{
		std::string displayMethodName;
		if ((lua["displayMethod"] >> displayMethodName).good()) {
			displayMethod = std::find(displayMethodNames.begin(), displayMethodNames.end(), displayMethodName) - displayMethodNames.begin();
		}
	}
	lua["displayScale"] >> displayScale;
	lua["useGravity"] >> useGravity;

	Super::init();


	//gradient texture
	{
		glGenTextures(1, &gradientTex);
		glBindTexture(GL_TEXTURE_2D, gradientTex);

		float colors[][3] = {
			{0, 0, 0},
			{0, 0, .5},
			{1, .5, 0},
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

		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, 1, 0, GL_RGB, GL_UNSIGNED_BYTE, data);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glBindTexture(GL_TEXTURE_2D, 0);
	}

	gradientTexMem = cl::ImageGL(context, CL_MEM_READ_ONLY, GL_TEXTURE_2D, 0, gradientTex);

	//read in the initial state
	std::vector<real4> stateVec(size.s[0] * size.s[1]);
	{
		if (!lua["initState"].isFunction()) throw Common::Exception() << "expected initState function";
		
		std::cout << "initializing..." << std::endl;
		real4* state = &stateVec[0];	
		int index[3];
		for (index[1] = 0; index[1] < size.s[1]; ++index[1]) {
			for (index[0] = 0; index[0] < size.s[0]; ++index[0], ++state) {
				real2 x;
				x.s[0] = real(xmax.s[0] - xmin.s[0]) * real(index[0]) / real(size.s[0]) + real(xmin.s[0]);
				x.s[1] = real(xmax.s[1] - xmin.s[1]) * real(index[1]) / real(size.s[1]) + real(xmin.s[1]);
				*state = lua["initState"].call<real4>(x);
			}
		}
		std::cout << "...done" << std::endl;
	}

	//construct the solver
	if (solverName == "Burgers") {
		solver = std::make_shared<BurgersSolver2D>(*this, stateVec);
	} else if (solverName == "Roe") {
		solver = std::make_shared<RoeSolver2D>(*this, stateVec);
	} else {
		throw Common::Exception() << "unknown solver " << solverName;
	}
	
	int err = glGetError();
	if (err) throw Common::Exception() << "GL error " << err;

	std::cout << "Success!" << std::endl;
}

void HydroGPUApp::shutdown() {
	glDeleteTextures(1, &gradientTex);
}

void HydroGPUApp::resize(int width, int height) {
	Super::resize(width, height);	//viewport
	screenSize = Tensor::Vector<int,2>(width, height);
	aspectRatio = (float)screenSize(0) / (float)screenSize(1);
	solver->resize();
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
		solver->addDrop();
	}
	
	if (doUpdate) {
		solver->update();
		if (doUpdate == 2) doUpdate = 0;
	}

	solver->display();
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
						solver->mouseZoom(dy);
					} 
				} else {
					if (dx || dy) {
						solver->mousePan(dx, dy);
					}
				}
			}
			int x = event.motion.x;
			int y = event.motion.y;
			solver->mouseMove(x, y, dx, dy);
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
				displayScale *= .5;
			} else {
				displayScale *= 2.;
			}
			std::cout << "displayScale " << displayScale << std::endl;
		} else if (event.key.keysym.sym == SDLK_d) {
			if (shiftDown) {
				displayMethod = (displayMethod + NUM_DISPLAY_METHODS - 1) % NUM_DISPLAY_METHODS;
			} else {
				displayMethod = (displayMethod + 1) % NUM_DISPLAY_METHODS;
			}
			std::cout << "display " << displayMethodNames[displayMethod] << std::endl;
		} else if (event.key.keysym.sym == SDLK_b) {
			if (shiftDown) {
				boundaryMethods(0) = (boundaryMethods(0) + NUM_BOUNDARY_METHODS - 1) % NUM_BOUNDARY_METHODS;
			} else {
				boundaryMethods(0) = (boundaryMethods(0) + 1) % NUM_BOUNDARY_METHODS;
			}
			for (int i = 1; i < 3; ++i) {
				boundaryMethods(i) = boundaryMethods(0);
			}
			std::cout << "boundary " << boundaryMethodNames[boundaryMethods(0)] << std::endl;
		} else if (event.key.keysym.sym == SDLK_u) {
			if (doUpdate) {
				doUpdate = 0;
			} else {
				if (shiftDown) {
					doUpdate = 2;
				} else {
					doUpdate = 1;
				}
			}
		}
		break;
	}
}

GLAPP_MAIN(HydroGPUApp)

