#include "HydroGPU/HydroGPUApp.h"
#include "HydroGPU/EulerHLL.h"
#include "HydroGPU/EulerBurgers.h"
#include "HydroGPU/EulerRoe.h"
#include "HydroGPU/SRHDRoe.h"
#include "HydroGPU/MHDRoe.h"
#include "HydroGPU/ADMRoe.h"
#include "Profiler/Profiler.h"
#include "Common/Exception.h"
#include "Common/File.h"
#include "Common/Macros.h"
#include <SDL2/SDL.h>
#include <OpenGL/gl.h>
#include <OpenGL/OpenGL.h>
#include <iostream>

//have to keep these updated with HydroGPU/Shared/Common.h

HydroGPUApp::HydroGPUApp()
: Super()
, gradientTex(GLuint())
, configFilename("config.lua")
, solverName("EulerBurgers")
, dim(0)
, doUpdate(1)
, maxFrames(-1)
, currentFrame(0)
, useFixedDT(false)
, fixedDT(.001f)
, cfl(.5f)
, displayMethod(0)
, displayScale(2.f)
, useGravity(false)
, gaussSeidelMaxIter(20)
, showVelocityField(true)
, velocityFieldResolution(16)
, velocityFieldScale(.125f)
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
	//config before Super::init so we can provide it 'useGPU'
	std::cout << "loading config file " << configFilename << std::endl;
	lua.loadFile(configFilename);
	if (!configString.empty()) {
		std::cout << "loading config string " << configString << std::endl;
		lua.loadString(configString);
	}
	
	lua.ref()["useGPU"] >> useGPU;
	for (int i = 0; i < 3; ++i) {
		if (!lua.ref()["size"].isNil()) lua.ref()["size"][i+1] >> size.s[i];
		if (!lua.ref()["xmin"].isNil()) lua.ref()["xmin"][i+1] >> xmin.s[i];
		if (!lua.ref()["xmax"].isNil()) lua.ref()["xmax"][i+1] >> xmax.s[i];
	}
	
	std::vector<std::string> boundaryMethodNames(3);
	for (int i = 0; i < 3; ++i) {
		if (!lua.ref()["boundaryMethods"].isNil()) {
			lua.ref()["boundaryMethods"][i+1] >> boundaryMethodNames[i];
		}
	}
	
	lua.ref()["maxFrames"] >> maxFrames;
	lua.ref()["showTimestep"] >> showTimestep;
	lua.ref()["solverName"] >> solverName;
	lua.ref()["useFixedDT"] >> useFixedDT;
	lua.ref()["fixedDT"] >> fixedDT;
	lua.ref()["cfl"] >> cfl;
	lua.ref()["displayScale"] >> displayScale;
	lua.ref()["useGravity"] >> useGravity;
	lua.ref()["gaussSeidelMaxIter"] >> gaussSeidelMaxIter;

	std::string displayMethodName;
	lua.ref()["displayMethod"] >> displayMethodName;

	lua.ref()["showVelocityField"] >> showVelocityField;
	lua.ref()["velocityFieldResolution"] >> velocityFieldResolution;
	lua.ref()["velocityFieldScale"] >> velocityFieldScale;

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
	std::cout << "solverName " << solverName << std::endl;
	if (solverName == "EulerBurgers") {
		solver = std::make_shared<EulerBurgers>(*this);
	} else if (solverName == "EulerHLL") {
		solver = std::make_shared<EulerHLL>(*this);
	} else if (solverName == "EulerRoe") {
		solver = std::make_shared<EulerRoe>(*this);
	} else if (solverName == "SRHDRoe") {
		solver = std::make_shared<SRHDRoe>(*this);	//broken
	} else if (solverName == "MHDRoe") {
		solver = std::make_shared<MHDRoe>(*this);	//broken
	} else if (solverName == "ADMRoe") {
		solver = std::make_shared<ADMRoe>(*this);
	} else {
		throw Common::Exception() << "unknown solver " << solverName;
	}
	solver->init();	//..now that the vtable is in place
	solver->resetState();

	displayMethod = std::find(
		solver->equation->displayMethods.begin(), 
		solver->equation->displayMethods.end(),
		displayMethodName) 
		- solver->equation->displayMethods.begin();
	if (displayMethod == solver->equation->displayMethods.size()) {
		throw Common::Exception() << "couldn't interpret display method " << displayMethodName;
	}

	for (int i = 0; i < 3; ++i) {
		if (boundaryMethodNames[i].empty()) continue;
		boundaryMethods(i) = std::find(
			solver->equation->boundaryMethods.begin(), 
			solver->equation->boundaryMethods.end(), 
			boundaryMethodNames[i]) 
			- solver->equation->boundaryMethods.begin();
		if (boundaryMethods(i) == solver->equation->boundaryMethods.size()) {
			throw Common::Exception() << "couldn't interpret boundary method " << boundaryMethodNames[i];
		}
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
				displayMethod = (displayMethod + solver->equation->displayMethods.size() - 1) % solver->equation->displayMethods.size();
			} else {
				displayMethod = (displayMethod + 1) % solver->equation->displayMethods.size();
			}
			std::cout << "display " << solver->equation->displayMethods[displayMethod] << std::endl;
		} else if (event.key.keysym.sym == SDLK_b) {
			if (shiftDown) {
				boundaryMethods(0) = (boundaryMethods(0) + solver->equation->boundaryMethods.size() - 1) % solver->equation->boundaryMethods.size();
			} else {
				boundaryMethods(0) = (boundaryMethods(0) + 1) % solver->equation->boundaryMethods.size();
			}
			for (int i = 1; i < 3; ++i) {
				boundaryMethods(i) = boundaryMethods(0);
			}
			std::cout << "boundary " << solver->equation->boundaryMethods[boundaryMethods(0)] << std::endl;
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
			solver->resetState();
		} else if (event.key.keysym.sym == SDLK_t) {
			showTimestep = !showTimestep;
		} else if (event.key.keysym.sym == SDLK_v) {
			showVelocityField = !showVelocityField;
			std::cout << "velocity field " << (showVelocityField ? "enabled" : "disabled") << std::endl;
		} else if (event.key.keysym.sym == SDLK_c) {
			if (shiftDown) {
				velocityFieldScale *= .5f;
			} else {
				velocityFieldScale *= 2.f;
			}
		}
		break;
	}
}

GLAPP_MAIN(HydroGPUApp)

