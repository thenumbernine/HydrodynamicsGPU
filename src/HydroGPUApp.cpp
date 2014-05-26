#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>

#include <string>
#include <fstream>
#include <iostream>
#include <vector>
#include <algorithm>

#include <SDL2/SDL.h>

//mac osx CL & GL stuff: 
#include <OpenCL/opencl.h>
#include <OpenGL/gl.h>
#include <OpenGL/OpenGL.h>

#include "Common/Finally.h"
#include "Common/Exception.h"
#include "Image/System.h"
#include "GLApp/GLApp.h" 

//OpenCL shared header
#include "roe_cell.h"

#define numberof(x)	(sizeof(x)/sizeof((x)[0]))
/*
kernels needed for Roe solver:
1) calc CFL <- reduce
2) calc eigen decomposition
	reads:	solid, q
	writes:	eigenvalues, eigenvectors, eigenvectorsInverse
3) calculate delta q tilde
	reads: solid, q
	writes:	deltaQTilde
4) calculate r tilde
	reads: deltaQTilde, eigenvalues
	writes: rTilde
5) calculate flux
	reads: solid, q, eigenvalues, eigenvectors, eigenvectorsInverse
	writes: flux
6) calculate dx/dt
	reads: q, flux, x
	writes: dx/dt
7) integrate
	reads: dx/dt
	writes: temp registers, q, etc

kernels needed for Burgers solver:
1) calc CFL	<- reduce
2) apply boundary	(or not since we will have already)
3) integrate flux
4) apply boundary
5) integrate pressure
6) apply boundary	(or not, since we will be again soon)
*/

std::string readFile(std::string filename) {
	std::ifstream f(filename);
	f.seekg(0, f.end);
	size_t len = f.tellg();
	f.seekg(0, f.beg);
	char *buf = new char[len];
	Finally finally([&](){ delete[] buf; });
	f.read(buf, len);
	return std::string(buf, len);
}

#define frand() ((double)rand() / (double)RAND_MAX)
#define crand()	(frand() * 2. - 1.)

struct HydroGPUApp : public GLApp {
	GLuint fluidTex;
	bool leftButtonDown;
	bool leftShiftDown;
	bool rightShiftDown;
	float viewZoom;
	float viewPosX;
	float viewPosY;

	HydroGPUApp()
	: GLApp()
	, fluidTex(0)
	, leftButtonDown(false)
	, leftShiftDown(false)
	, rightShiftDown(false)
	, viewZoom(1.f)
	, viewPosX(0.f)
	, viewPosY(0.f)
	{}
	
	virtual void init();
	virtual void shutdown();
	virtual void resize(int width, int height);
	virtual void update();
	virtual void sdlEvent(SDL_Event &event);
};

void HydroGPUApp::init() {
	GLApp::init();

	int err;
	  
	size_t global[DIM];
	  
	std::string kernelSource = readFile("res/roe_integrate_flux.cl");

	real noise = real(.01);
	int size[DIM] = {256, 256};
	real xmin[DIM] = {-.5, -.5};
	real xmax[DIM] = {.5, .5};
	unsigned int count = size[0] * size[1];
	
	std::vector<Cell> cells(count);
	{
		int index[DIM];
		
		Cell *cell = &cells[0];
		//for (index[2] = 0; index[2] < size[2]; ++index[2]) {
			for (index[1] = 0; index[1] < size[1]; ++index[1]) {
				for (index[0] = 0; index[0] < size[0]; ++index[0], ++cell) {
					bool lhs = true;
					for (int n = 0; n < DIM; ++n) {
						cell->x.s[n] = real(xmax[n] - xmin[n]) * real(index[n]) / real(size[n]) + real(xmin[n]);
						if (cell->x.s[n] > real(.3) * real(xmax[n]) + real(.7) * real(xmin[n])) {
							lhs = false;
						}
					}

					for (int m = 0; m < DIM; ++m) {
						cell->interfaces[m].solid = false;
						for (int n = 0; n < DIM; ++n) {
							cell->interfaces[m].x.s[n] = cell->x.s[n];
							if (m == n) {
								cell->interfaces[m].x.s[n] -= real(xmax[n] - xmin[n]) * real(.5) / real(size[n]);
							}
						}
					}

					//sod init
					real density = lhs ? 1. : .1;
					real velocity[DIM];
					real energyKinetic = real();
					for (int n = 0; n < DIM; ++n) {
						velocity[n] = crand() * noise;
						energyKinetic += velocity[n] * velocity[n];
					}
					energyKinetic *= real(.5);
					real energyThermal = 1.;
					real energyTotal = energyKinetic + energyThermal;

					cell->q.s[0] = density;
					for (int n = 0; n < DIM; ++n) {
						cell->q.s[n+1] = density * velocity[n];
					}
					cell->q.s[DIM+1] = density * energyTotal;
				}
			}
		//}
	}

	int gpu = 1;	//whether we want to request the GPU or CPU
	
	cl_uint numPlatforms = 0;
	err = clGetPlatformIDs(0, NULL, &numPlatforms);
	if (err != CL_SUCCESS || numPlatforms == 0) throw Exception() << "failed to query number of CL platforms.  got error " << err;
	
	std::vector<cl_platform_id> platformIDs(numPlatforms);
	err = clGetPlatformIDs(numPlatforms, &platformIDs[0], NULL);
	if (err != CL_SUCCESS) throw Exception() << "failed to query CL platforms.  got error " << err;
 	
	cl_platform_id platformID = platformIDs[0];

	cl_uint numDevices = 0;
	err = clGetDeviceIDs(platformID, gpu ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU, 0, NULL, &numDevices);
	if (err != CL_SUCCESS || numDevices == 0) throw Exception() << "failed to query number of CL devices.  got error " << err;

	std::vector<cl_device_id> deviceIDs(numDevices);

	err = clGetDeviceIDs(platformID, gpu ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU, numDevices, &deviceIDs[0], NULL);
	if (err != CL_SUCCESS) throw Exception() << "Error: Failed to create a device group!";

	std::vector<cl_device_id>::iterator deviceIter =
		std::find_if(deviceIDs.begin(), deviceIDs.end(), [&](cl_device_id deviceID)
	{
		size_t param_value_size_ret = 0;
		err = clGetDeviceInfo(deviceID, CL_DEVICE_EXTENSIONS, 0, NULL, &param_value_size_ret );
		if (err != CL_SUCCESS) throw Exception() << "clGetDeviceInfo failed for device " << deviceID << " with error " << err;		
	
		std::string param_value(param_value_size_ret, '\0');
		err = clGetDeviceInfo(deviceID, CL_DEVICE_EXTENSIONS, param_value_size_ret, (void*)param_value.c_str(), NULL );
		if (err != CL_SUCCESS) throw Exception() << "clGetDeivceInfo failed for device " << deviceID << " with error " << err;

		std::vector<std::string> caps;
		std::istringstream iss(param_value);
		std::copy(std::istream_iterator<std::string>(iss), std::istream_iterator<std::string>(), std::back_inserter<std::vector<std::string>>(caps));

		std::vector<std::string>::iterator extension = 
			std::find_if(caps.begin(), caps.end(), [&](const std::string &s)
		{
			return s == std::string("cl_khr_gl_sharing") 
				|| s == std::string("cl_APPLE_gl_sharing");
		});
	 
		return extension != caps.end();
	});
	if (deviceIter == deviceIDs.end()) throw Exception() << "failed to find a device with cap cl_khr_gl_sharing";
	cl_device_id deviceID = *deviceIter;

#if PLATFORM_OSX
	CGLContextObj kCGLContext = CGLGetCurrentContext();	// GL Context
	CGLShareGroupObj kCGLShareGroup = CGLGetShareGroup(kCGLContext); // Share Group
	cl_context_properties properties[] = {
		CL_CONTEXT_PROPERTY_USE_CGL_SHAREGROUP_APPLE, (cl_context_properties)kCGLShareGroup,
		CL_CONTEXT_PLATFORM, (cl_context_properties)platformID,
		0
	};
#endif
#if PLATFORM_WINDOWS
	cl_context_properties properties[] = {
		CL_GL_CONTEXT_KHR, (cl_context_properties) wglGetCurrentContext(), // HGLRC handle
		CL_WGL_HDC_KHR, (cl_context_properties) wglGetCurrentDC(), // HDC handle
		CL_CONTEXT_PLATFORM, (cl_context_properties)cpPlatform, 
		0
	};	
#endif

	cl_context context = clCreateContext(properties, 1, &deviceID, NULL, NULL, &err);
	if (!context) throw Exception() << "Error: Failed to create a compute context!";
	Finally contextFinally([&](){ clReleaseContext(context); });
 
	cl_command_queue commands = clCreateCommandQueue(context, deviceID, 0, &err);
	if (!commands) throw Exception() << "Error: Failed to create a command queue!";
	Finally commandsFinally([&](){ clReleaseCommandQueue(commands); });

	//get a texture going for visualizing the output
	glGenTextures(1, &fluidTex);
	glBindTexture(GL_TEXTURE_2D, fluidTex);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexImage2D(GL_TEXTURE_2D, 0, 4, size[0], size[1], 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	glBindTexture(GL_TEXTURE_2D, 0);
	if ((err = glGetError()) != 0) throw Exception() << "failed to create GL texture.  got error " << err;
	
	cl_mem texMem = clCreateFromGLTexture(context, CL_MEM_WRITE_ONLY, GL_TEXTURE_2D, 0, fluidTex, &err);
	if (!texMem) throw Exception() << "failed to create CL memory from GL texture.  got error " << err;

	const char *kernelSourcePtr = kernelSource.c_str();
	cl_program program = clCreateProgramWithSource(context, 1, (const char **) &kernelSourcePtr, NULL, &err);
	if (!program) throw Exception() << "Error: Failed to create compute program!";
	Finally programFinally([&](){ clReleaseProgram(program); });
 
	err = clBuildProgram(program, 0, NULL, "-I res/include", NULL, NULL);
	if (err != CL_SUCCESS) {
		size_t len;
		char buffer[2048];
 
		std::cout << "Error: Failed to build program executable!\n" << std::endl;
		clGetProgramBuildInfo(program, deviceID, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
		std::cout << buffer << std::endl;
		exit(1);
	}
 
	cl_mem cellsMem = clCreateBuffer(context,  CL_MEM_READ_WRITE,  sizeof(Cell) * count, NULL, NULL);
	if (!cellsMem) throw Exception() << "Error: Failed to allocate device memory!";
	Finally cellsMemFinally([&](){ clReleaseMemObject(cellsMem); });

	err = clEnqueueWriteBuffer(commands, cellsMem, CL_TRUE, 0, sizeof(Cell) * count, &cells[0], 0, NULL, NULL);
	if (err != CL_SUCCESS) {
		std::cout << "Error: Failed to write to source array!\n" << std::endl;
		exit(1);
	}
 
	for (int n = 0; n < DIM; ++n) {
		global[n] = size[n];
	}

	cl_kernel calcEigenDecompositionKernel = clCreateKernel(program, "calcEigenDecomposition", &err);
	if (!calcEigenDecompositionKernel || err != CL_SUCCESS) throw Exception() << "failed to create kernel";
	Finally k1Finally([&](){ clReleaseKernel(calcEigenDecompositionKernel); });
	
	cl_kernel calcDeltaQTildeKernel = clCreateKernel(program, "calcDeltaQTilde", &err);
	if (!calcDeltaQTildeKernel || err != CL_SUCCESS) throw Exception() << "failed to create kernel";
	Finally k2Finally([&](){ clReleaseKernel(calcDeltaQTildeKernel ); });
	
	cl_kernel calcRTildeKernel = clCreateKernel(program, "calcRTilde", &err);
	if (!calcRTildeKernel || err != CL_SUCCESS) throw Exception() << "failed to create kernel";
	Finally k3Finally([&](){ clReleaseKernel(calcRTildeKernel ); });

	cl_kernel calcFluxKernel = clCreateKernel(program, "calcFlux", &err);
	if (!calcFluxKernel || err != CL_SUCCESS) throw Exception() << "failed to create kernel";
	Finally k4Finally([&](){ clReleaseKernel(calcFluxKernel ); });
	
	cl_kernel updateStateKernel = clCreateKernel(program, "updateState", &err);
	if (!updateStateKernel || err != CL_SUCCESS) throw Exception() << "failed to create kernel";
	Finally k5Finally([&](){ clReleaseKernel(updateStateKernel); });

	cl_kernel copyToTexKernel = clCreateKernel(program, "copyToTex", &err);
	if (!copyToTexKernel || err != CL_SUCCESS) throw Exception() << "failed to create kernel";
	Finally k6Finally([&](){ clReleaseKernel(copyToTexKernel); });

	cl_kernel* kernels[] = {
		&calcEigenDecompositionKernel,
		&calcDeltaQTildeKernel,
		&calcRTildeKernel,
		&calcFluxKernel,
		&updateStateKernel,
	};
	std::for_each(kernels, kernels + numberof(kernels), [&](cl_kernel* kernel) {
		err = 0;
		err  = clSetKernelArg(*kernel, 0, sizeof(cl_mem), &cellsMem);
		err |= clSetKernelArg(*kernel, 1, sizeof(cl_uint2), &size[0]);
		if (err != CL_SUCCESS) throw Exception() << "Error: Failed to set kernel arguments! " << err;
	});
	
	real dx[DIM];
	for (int i = 0; i < DIM; ++i) {
		dx[i] = (xmax[i] - xmin[i]) / (float)size[i];
	}
	real dt = .01;
	real dt_dx[DIM];
	for (int i = 0; i < DIM; ++i) {
		dt_dx[i] = dt / dx[i];
	}
	err = clSetKernelArg(calcFluxKernel, 2, DIM * sizeof(real), dt_dx);
	if (err != CL_SUCCESS) throw Exception() << "Error: Failed to set kernel arguments! " << err;

	err = clSetKernelArg(updateStateKernel, 2, DIM * sizeof(real), dt_dx);
	if (err != CL_SUCCESS) throw Exception() << "Error: Failed to set kernel arguments! " << err;

	err = 0;
	err  = clSetKernelArg(copyToTexKernel, 0, sizeof(cl_mem), &cellsMem);
	err |= clSetKernelArg(copyToTexKernel, 1, sizeof(cl_uint2), &size[0]);
	err |= clSetKernelArg(copyToTexKernel, 2, sizeof(cl_mem), &texMem);
	if (err != CL_SUCCESS) throw Exception() << "Error: Failed to set kernel arguments! " << err;

/*
	why isn't this working?
	// Get the maximum work group size for executing the kernel on the device
	//
	err = clGetKernelWorkGroupInfo(kernel, deviceID, CL_KERNEL_WORK_GROUP_SIZE, 2 * sizeof(local), local, NULL);
	if (err != CL_SUCCESS) {
		std::cout << "Error: Failed to retrieve kernel work group info! " << err << std::endl;
		exit(1);
	}
*/	//manually provide it in the mean time: 
	size_t local[DIM] = {16, 16};

	std::for_each(kernels, kernels + numberof(kernels), [&](cl_kernel* kernel) {
		err = clEnqueueNDRangeKernel(commands, *kernel, 2, NULL, global, local, 0, NULL, NULL);
		if (err) throw Exception() << "Error: Failed to execute kernel!";
	});

	glFlush();
	glFinish();
	clEnqueueAcquireGLObjects(commands, 1, &texMem, 0, 0, 0);
	
	err = clEnqueueNDRangeKernel(commands, copyToTexKernel, 2, NULL, global, local, 0, NULL, NULL);
	if (err) throw Exception() << "Error: Failed to execute kernel!";

	clEnqueueReleaseGLObjects(commands, 1, &texMem, 0, 0, 0);
	clFlush(commands);
	clFinish(commands);

#if 0
	err = clEnqueueReadBuffer( commands, cellsMem, CL_TRUE, 0, sizeof(Cell) * count, &cells[0], 0, NULL, NULL );  
	if (err != CL_SUCCESS) throw Exception() << "Error: Failed to read cellsMem array! " << err;
#endif

	std::cout << "Success!" << std::endl;
}

void HydroGPUApp::shutdown() {
}

void HydroGPUApp::resize(int width, int height) {
	GLApp::resize(width, height);	//viewport
	float aspectRatio = (float)width / (float)height;
	glOrtho(-aspectRatio, aspectRatio, -1., 1., -1., 1.);
}

void HydroGPUApp::update() {
	GLApp::update();	//glclear 
	
	glPushMatrix();
	glTranslatef(-viewPosX, -viewPosY, 0);
	glScalef(viewZoom, viewZoom, viewZoom);
	glEnable(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, fluidTex);
	glBegin(GL_QUADS);
	glTexCoord2f(0,0); glVertex2f(-.5f,-.5f);
	glTexCoord2f(1,0); glVertex2f(.5f,-.5f);
	glTexCoord2f(1,1); glVertex2f(.5f,.5f);
	glTexCoord2f(0,1); glVertex2f(-.5f,.5f);
	glEnd();
	glBindTexture(GL_TEXTURE_2D, 0);
	glPopMatrix();

	int err = 0;
	if ((err = glGetError())) std::cout << "error " << err << std::endl;
}

void HydroGPUApp::sdlEvent(SDL_Event &event) {
	bool shiftDown = leftShiftDown | rightShiftDown;

	switch (event.type) {
	case SDL_MOUSEMOTION:
		{	
			int dx = event.motion.xrel;
			int dy = event.motion.yrel;
			if (leftButtonDown) {
				if (shiftDown) {
					if (dy) {
						viewZoom *= exp((float)dy * -.03f); 
					} 
				} else {
					if (dx || dy) {
						viewPosX -= (float)dx * 0.01f;
						viewPosY += (float)dy * 0.01f;
					}
				}
			}
		}
		break;
	case SDL_MOUSEBUTTONDOWN:
		if (event.button.button == SDL_BUTTON_LEFT) {
			leftButtonDown = true;
		}
		break;
	case SDL_MOUSEBUTTONUP:
		if (event.button.button == SDL_BUTTON_LEFT) {
			leftButtonDown = false;
		}
		break;
	case SDL_KEYDOWN:
		if (event.key.keysym.sym == SDLK_LSHIFT) {
			leftShiftDown = true;
		} else if (event.key.keysym.sym == SDLK_RSHIFT) {
			rightShiftDown = true;
		}
		break;
	case SDL_KEYUP:
		if (event.key.keysym.sym == SDLK_LSHIFT) {
			leftShiftDown = false;
		} else if (event.key.keysym.sym == SDLK_RSHIFT) {
			rightShiftDown = false;
		}
		break;
	}
}

GLAPP_MAIN(HydroGPUApp)

