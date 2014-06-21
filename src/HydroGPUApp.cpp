#include "HydroGPU/HydroGPUApp.h"
#include "HydroGPU/RoeSolver2D.h"
#include "HydroGPU/HLLSolver2D.h"
#include "HydroGPU/BurgersSolver3D.h"
#include "HydroGPU/RoeSolver3D.h"
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
template<> void fromC<real3> (lua_State *L, const real3& value) { 
	lua_newtable(L);
	int t = lua_gettop(L);
	for (int i = 0; i < 3; ++i) {
		lua_pushnumber(L, value.s[i]);
		lua_rawseti(L, t, i+1);
	}
}

template<> real8 toC<real8>(lua_State *L, int loc) {
	real8 result; 
	for (int i = 0; i < 8; ++i) { 
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
, dim(0)
, doUpdate(1)
, maxFrames(-1)
, currentFrame(0)
, useFixedDT(false)
, fixedDT(.001f)
, cfl(.5f)
, displayMethod(DISPLAY_DENSITY)
, displayScale(2.f)
, boundaryMethods(BOUNDARY_PERIODIC, BOUNDARY_PERIODIC, BOUNDARY_PERIODIC)
, useGravity(false)
, gamma(1.4)
, leftButtonDown(false)
, rightButtonDown(false)
, leftShiftDown(false)
, rightShiftDown(false)
, leftGuiDown(false)
, rightGuiDown(false)
, showTimestep(false)
{
	for (int i = 0; i < 3; ++i) {
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
	//config before Super::init so we can provide it 'useGPU'
	std::cout << "loading config file " << configFilename << std::endl;
	lua.loadFile(configFilename);
	if (!configString.empty()) {
		std::cout << "loading config string " << configString << std::endl;
		lua.loadString(configString);
	}
	
	lua["useGPU"] >> useGPU;
	for (int i = 0; i < 3; ++i) {
		if (!lua["size"].isNil()) lua["size"][i+1] >> size.s[i];
		if (!lua["xmin"].isNil()) lua["xmin"][i+1] >> xmin.s[i];
		if (!lua["xmax"].isNil()) lua["xmax"][i+1] >> xmax.s[i];
		if (!lua["boundaryMethods"].isNil()) {
			std::string boundaryMethodName;
			if ((lua["boundaryMethods"][i+1] >> boundaryMethodName).good()) {
				boundaryMethods(i) = std::find(boundaryMethodNames.begin(), boundaryMethodNames.end(), boundaryMethodName) - boundaryMethodNames.begin();
				if (boundaryMethods(i) == NUM_BOUNDARY_METHODS) throw Common::Exception() << "couldn't interpret boundary method " << boundaryMethodName;
			}
		}
	}
	lua["maxFrames"] >> maxFrames;
	lua["showTimestep"] >> showTimestep;
	lua["solverName"] >> solverName;
	lua["useFixedDT"] >> useFixedDT;
	lua["fixedDT"] >> fixedDT;
	lua["cfl"] >> cfl;
	lua["gamma"] >> gamma;
	{
		std::string displayMethodName;
		if ((lua["displayMethod"] >> displayMethodName).good()) {
			displayMethod = std::find(displayMethodNames.begin(), displayMethodNames.end(), displayMethodName) - displayMethodNames.begin();
			if (displayMethod == NUM_DISPLAY_METHODS) throw Common::Exception() << "couldn't interpret display method " << displayMethodName;
		}
	}
	lua["displayScale"] >> displayScale;
	lua["useGravity"] >> useGravity;

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

	glEnable(GL_DEPTH_TEST);

	//gradient texture
	{
		glGenTextures(1, &gradientTex);
		glBindTexture(GL_TEXTURE_1D, gradientTex);

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

		glTexImage1D(GL_TEXTURE_1D, 0, GL_RGB, width, 0, GL_RGB, GL_UNSIGNED_BYTE, data);
		glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glBindTexture(GL_TEXTURE_1D, 0);
	}

	gradientTexMem = cl::ImageGL(context, CL_MEM_READ_ONLY, GL_TEXTURE_1D, 0, gradientTex);

	//construct the solver
	if (solverName == "Burgers") {
		switch (dim) {
		case 1:
		case 2:
		case 3:
			solver = std::make_shared<BurgersSolver3D>(*this);
			break;
		default:
			throw Common::Exception() << "solver " << solverName << " can't handle dim " << dim;
		}
	} else if (solverName == "Roe") {
		switch (dim) {
		case 1:
		case 2:
			solver = std::make_shared<RoeSolver2D>(*this);
			break;
		case 3:
			solver = std::make_shared<RoeSolver3D>(*this);	//broken
			break;
		default:
			throw Common::Exception() << "solver " << solverName << " can't handle dim " << dim;
		}
	} else if (solverName == "HLL") {
		switch (dim) {
		case 1:
		case 2:
			solver = std::make_shared<HLLSolver2D>(*this);
			break;
		default:
			throw Common::Exception() << "solver " << solverName << " can't handle dim " << dim;
		}
	} else {
		throw Common::Exception() << "unknown solver " << solverName;
	}

	resetState();

	int err = glGetError();
	if (err) throw Common::Exception() << "GL error " << err;

	std::cout << "Success!" << std::endl;
}

void HydroGPUApp::shutdown() {
	glDeleteTextures(1, &gradientTex);
}

void HydroGPUApp::resetState() {
	std::vector<real8> stateVec(size.s[0] * size.s[1] * size.s[2]);
		
	if (!lua["initState"].isFunction()) throw Common::Exception() << "expected initState function";
	
	std::cout << "initializing..." << std::endl;
	real8* state = &stateVec[0];	
	int index[3];
	for (index[2] = 0; index[2] < size.s[2]; ++index[2]) {
		for (index[1] = 0; index[1] < size.s[1]; ++index[1]) {
			for (index[0] = 0; index[0] < size.s[0]; ++index[0], ++state) {
				real3 pos;
				for (int i = 0; i < 3; ++i) {
					pos.s[i] = real(xmax.s[i] - xmin.s[i]) * (real(index[i]) + .5) / real(size.s[i]) + real(xmin.s[i]);
				}
				*state = lua["initState"].call<real8>(pos);
			}
		}
	}
	std::cout << "...done" << std::endl;

	solver->resetState(stateVec);
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
		} else if (event.key.keysym.sym == SDLK_r) {
			resetState();
		} else if (event.key.keysym.sym == SDLK_t) {
			showTimestep = !showTimestep;
		}
		break;
	}
}

GLAPP_MAIN(HydroGPUApp)

