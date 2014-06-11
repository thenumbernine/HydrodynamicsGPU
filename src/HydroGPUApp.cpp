#include "HydroGPU/HydroGPUApp.h"
#include "HydroGPU/RoeSolver.h"
#include "HydroGPU/BurgersSolver.h"
#include "Profiler/Profiler.h"
#include "Common/Exception.h"
#include "Common/Macros.h"
#include <SDL2/SDL.h>
#include <OpenGL/gl.h>
#include <OpenGL/OpenGL.h>
#include <iostream>

HydroGPUArgs::HydroGPUArgs()
: solverName("Burgers")
{}

HydroGPUApp::HydroGPUApp()
: Super()
, fluidTex(GLuint())
, gradientTex(GLuint())
, doUpdate(1)
, maxFrames(-1)
, currentFrame(0)
, leftButtonDown(false)
, rightButtonDown(false)
, leftShiftDown(false)
, rightShiftDown(false)
, leftGuiDown(false)
, rightGuiDown(false)
, viewZoom(1.f)
, useFixedDT(false)
, fixedDT(.001f)
, cfl(.5f)
, displayMethod(DISPLAY_DENSITY)
, displayScale(2.f)
, boundaryMethod(BOUNDARY_REPEAT)
, useGravity(true)
{
	for (int i = 0; i < DIM; ++i) {
		size.s[i] = 512;
	}
}

int HydroGPUApp::main(std::vector<std::string> args) {
	for (int i = 1; i < args.size(); ++i) {
		if (args[i] == "--cpu") {
			useGPU = false;
			continue;
		}
		if (i < args.size()-1) {
			if (args[i] == "--frames") {
				maxFrames = std::stoi(args[++i]);
				continue;
			} else if (args[i] == "--solver") {
				hydroArgs.solverName = args[++i];
				continue;
			}
		}
		if (i < args.size()-DIM) {
			if (args[i] == "--size") {
				for (int k = 0; k < DIM; ++k) {
					size.s[k] = std::stoi(args[++i]);
				}
				continue;
			}
		}
		throw Common::Exception() << "got unknown cmdline argument: " << args[i];
	}
	return Super::main(args);
}

void HydroGPUApp::init() {
	Super::init();

	int err;
	  
	for (int n = 0; n < DIM; ++n) {
		xmin.s[n] = -.5f;
		xmax.s[n] = .5f;
	}
	
	//get a texture going for visualizing the output
	glGenTextures(1, &fluidTex);
	glBindTexture(GL_TEXTURE_2D, fluidTex);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexImage2D(GL_TEXTURE_2D, 0, 4, size.s[0], size.s[1], 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	glBindTexture(GL_TEXTURE_2D, 0);
	if ((err = glGetError()) != 0) throw Common::Exception() << "failed to create GL texture.  got error " << err;

	//hmm, my cl.hpp version only supports clCreateFromGLTexture2D, which is deprecated ... do I use the deprecated method, or do I stick with the C structures?
	// ... or do I look for a more up-to-date version of cl.hpp
	fluidTexMem = cl::ImageGL(context, CL_MEM_WRITE_ONLY, GL_TEXTURE_2D, 0, fluidTex);

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

	if (hydroArgs.solverName == "Burgers") {
		solver = std::make_shared<BurgersSolver>(*this);
	} else if (hydroArgs.solverName == "Roe") {
		solver = std::make_shared<RoeSolver>(*this);
	} else {
		throw Common::Exception() << "unknown solver " << hydroArgs.solverName;
	}
	
	err = glGetError();
	if (err) throw Common::Exception() << "GL error " << err;

	std::cout << "Success!" << std::endl;
}

void HydroGPUApp::shutdown() {
	glDeleteTextures(1, &fluidTex);
	glDeleteTextures(1, &gradientTex);
}

void HydroGPUApp::resize(int width, int height) {
	Super::resize(width, height);	//viewport
	screenSize = Tensor::Vector<int,2>(width, height);
	aspectRatio = (float)width / (float)height;
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(-aspectRatio *.5, aspectRatio * .5, -.5, .5, -1., 1.);
	glMatrixMode(GL_MODELVIEW);
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
		solver->addDrop(mousePos, mouseVel);
	}
	
	//CPU need to bind beforehand for roe/cpu to use it
	//GPU needs it unbound until after the update
	if (!useGPU) {
		glBindTexture(GL_TEXTURE_2D, fluidTex);
	}
	
	if (doUpdate) {
		solver->update();
		if (doUpdate == 2) doUpdate = 0;
	}
	
	glPushMatrix();
	glTranslatef(-viewPos(0), -viewPos(1), 0);
	glScalef(viewZoom, viewZoom, viewZoom);
	glBindTexture(GL_TEXTURE_2D, fluidTex);
	glEnable(GL_TEXTURE_2D);
	glBegin(GL_QUADS);
	glTexCoord2f(0,0); glVertex2f(-.5f,-.5f);
	glTexCoord2f(1,0); glVertex2f(.5f,-.5f);
	glTexCoord2f(1,1); glVertex2f(.5f,.5f);
	glTexCoord2f(0,1); glVertex2f(-.5f,.5f);
	glEnd();
	glBindTexture(GL_TEXTURE_2D, 0);
	glPopMatrix();

	{int err = glGetError();
	if (err) std::cout << "GL error " << err << " at " << __LINE__ << std::endl;}
PROFILE_END_FRAME();
}

void HydroGPUApp::sdlEvent(SDL_Event &event) {
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
						float scale = exp((float)dy * -.03f); 
						viewPos *= scale;
						viewZoom *= scale; 
					} 
				} else {
					if (dx || dy) {
						viewPos += Tensor::Vector<float,2>(-(float)dx * aspectRatio / (float)screenSize(0), (float)dy / (float)screenSize(1));
					}
				}
			}
		}
		{
			mousePos(0) = (float)event.motion.x / (float)screenSize(0) * (xmax.s[0] - xmin.s[0]) + xmin.s[0];
			mousePos(0) *= aspectRatio;	//only if xmin/xmax is symmetric. otehrwise more math required.
			mousePos(1) = (1.f - (float)event.motion.y / (float)screenSize(1)) * (xmax.s[1] - xmin.s[1]) + xmin.s[1];
			mousePos += viewPos;
			mousePos /= viewZoom;
			mouseVel(0) = (float)event.motion.xrel / (float)screenSize(0);
			mouseVel(1) = (float)event.motion.yrel / (float)screenSize(1);
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
		} else if (event.key.keysym.sym == SDLK_d) {
			if (shiftDown) {
				displayMethod = (displayMethod + NUM_DISPLAY_METHODS - 1) % NUM_DISPLAY_METHODS;
			} else {
				displayMethod = (displayMethod + 1) % NUM_DISPLAY_METHODS;
			}
			switch (displayMethod) {
			case DISPLAY_DENSITY:	std::cout << "display density" << std::endl; break;
			case DISPLAY_VELOCITY:	std::cout << "display velocity" << std::endl; break;
			case DISPLAY_PRESSURE:	std::cout << "display pressure" << std::endl; break;
			case DISPLAY_GRAVITY_POTENTIAL:	std::cout << "display gravity potential" << std::endl; break;
			default: std::cout << "unknown display" << std::endl; break;
			}
		} else if (event.key.keysym.sym == SDLK_b) {
			if (shiftDown) {
				boundaryMethod = (boundaryMethod + NUM_BOUNDARY_METHODS - 1) % NUM_BOUNDARY_METHODS;
			} else {
				boundaryMethod = (boundaryMethod + 1) % NUM_BOUNDARY_METHODS;
			}
			switch (boundaryMethod) {
			case BOUNDARY_REPEAT:	std::cout << "boundary repeat" << std::endl; break;
			case BOUNDARY_MIRROR:	std::cout << "boundary mirror" << std::endl; break;
			case BOUNDARY_FREEFLOW:	std::cout << "boundary freeflow" << std::endl; break;
			default: std::cout << "unknown boundary" << std::endl; break;
			}		
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

