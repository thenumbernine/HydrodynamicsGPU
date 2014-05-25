#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <OpenCL/opencl.h>

#include <string>
#include <fstream>
#include <iostream>
#include <vector>
#include <algorithm>

#include "Common/Finally.h"
#include "Common/Exception.h"
#include "Image/System.h"

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

int main(int argc, char** argv)
{
	int err;
	  
	size_t global[DIM];
 
	  
	std::string kernelSource = readFile("res/roe_integrate_flux.cl");

	real noise = real(.01);
	int size[DIM] = {1024, 1024};
	real xmin[DIM] = {.5, .5};
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
						if (cell->x.s[n] > real(.5) * real(xmax[n] + xmin[n])) {
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
#if 0

	// Connect to a compute device
	//
	int gpu = 1;
	cl_device_id deviceID;
	err = clGetDeviceIDs(NULL, gpu ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU, 1, &deviceID, NULL);
	if (err != CL_SUCCESS) throw Exception() << "Error: Failed to create a device group!";
  
	cl_context context = clCreateContext(0, 1, &deviceID, NULL, NULL, &err);
	if (!context) throw Exception() << "Error: Failed to create a compute context!";
	Finally([&](){ clReleaseContext(context); });
 
	cl_command_queue commands = clCreateCommandQueue(context, deviceID, 0, &err);
	if (!commands) throw Exception() << "Error: Failed to create a command commands!";
	Finally([&](){ clReleaseCommandQueue(commands); });
 
	const char *kernelSourcePtr = kernelSource.c_str();
	cl_program program = clCreateProgramWithSource(context, 1, (const char **) &kernelSourcePtr, NULL, &err);
	if (!program) throw Exception() << "Error: Failed to create compute program!";
	Finally([&](){ clReleaseProgram(program); });
 
	err = clBuildProgram(program, 0, NULL, "-I res/include", NULL, NULL);
	if (err != CL_SUCCESS) {
		size_t len;
		char buffer[2048];
 
		std::cout << "Error: Failed to build program executable!\n" << std::endl;
		clGetProgramBuildInfo(program, deviceID, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
		std::cout << buffer << std::endl;
		exit(1);
	}
 
	cl_mem cl_cells = clCreateBuffer(context,  CL_MEM_READ_WRITE,  sizeof(Cell) * count, NULL, NULL);
	if (!cl_cells) throw Exception() << "Error: Failed to allocate device memory!";
	Finally([&](){ clReleaseMemObject(cl_cells); });

	err = clEnqueueWriteBuffer(commands, cl_cells, CL_TRUE, 0, sizeof(Cell) * count, &cells[0], 0, NULL, NULL);
	if (err != CL_SUCCESS) {
		std::cout << "Error: Failed to write to source array!\n" << std::endl;
		exit(1);
	}
 
	for (int n = 0; n < DIM; ++n) {
		global[n] = size[n];
	}

	cl_kernel calcEigenDecompositionKernel = clCreateKernel(program, "calcEigenDecomposition", &err);
	if (!calcEigenDecompositionKernel || err != CL_SUCCESS) throw Exception() << "failed to create kernel";
	Finally([&](){ clReleaseKernel(calcEigenDecompositionKernel); });
	
	cl_kernel calcDeltaQTildeKernel = clCreateKernel(program, "calcDeltaQTilde", &err);
	if (!calcDeltaQTildeKernel || err != CL_SUCCESS) throw Exception() << "failed to create kernel";
	Finally([&](){ clReleaseKernel(calcDeltaQTildeKernel ); });
	
	cl_kernel calcRTildeKernel = clCreateKernel(program, "calcRTilde", &err);
	if (!calcRTildeKernel || err != CL_SUCCESS) throw Exception() << "failed to create kernel";
	Finally([&](){ clReleaseKernel(calcRTildeKernel ); });

	cl_kernel calcFluxKernel = clCreateKernel(program, "calcFlux", &err);
	if (!calcFluxKernel || err != CL_SUCCESS) throw Exception() << "failed to create kernel";
	Finally([&](){ clReleaseKernel(calcFluxKernel ); });
	
	cl_kernel updateStateKernel = clCreateKernel(program, "updateState", &err);
	if (!updateStateKernel || err != CL_SUCCESS) throw Exception() << "failed to create kernel";
	Finally([&](){ clReleaseKernel(updateStateKernel); });
	
	cl_kernel* kernels[] = {
		&calcEigenDecompositionKernel,
		&calcDeltaQTildeKernel,
		&calcRTildeKernel,
		&calcFluxKernel,
		&updateStateKernel,
	};
	std::for_each(kernels, kernels + numberof(kernels), [&](cl_kernel* kernel) {
		err = 0;
		err  = clSetKernelArg(*kernel, 0, sizeof(cl_mem), &cl_cells);
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
 
	clFinish(commands);

	err = clEnqueueReadBuffer( commands, cl_cells, CL_TRUE, 0, sizeof(Cell) * count, &cells[0], 0, NULL, NULL );  
	if (err != CL_SUCCESS) throw Exception() << "Error: Failed to read cl_cells array! " << err;
#endif

	//write out to an image or something
	Image::ImageType<> image(Vector<int,2>(size[0], size[1]), NULL);
	char *pixel = image.getDataType();
	Cell *cell = &cells[0];
	float colors[][3] = {
		{0, 0, .5},
		{1, 1, 0},
		{1, .5, 0},
		{1, 0, 0}
	};
	for (int j = 0; j < size[1]; ++j) {
		for (int i = 0; i < size[0]; ++i) {
			float f = cell->q.s[0]; ++cell;
			f *= 2.;	//value scale
			f *= (float)numberof(colors);	//pallete count
			int n = ((int)f) % numberof(colors);
			n = (n + numberof(colors)) % numberof(colors);
			float s = f - (float)n;
			float t = 1. - s;
			float *ca = colors[n];
			float *cb = colors[(n+1)%numberof(colors)];
			*pixel = (char)(255.f * (cb[0] * s + ca[0] * t)); ++pixel;
			*pixel = (char)(255.f * (cb[1] * s + ca[1] * t)); ++pixel;
			*pixel = (char)(255.f * (cb[2] * s + ca[2] * t)); ++pixel;
		}
	}
	Image::sys->save(&image, "output.png");

	std::cout << "Success!" << std::endl;

	return 0;
}
