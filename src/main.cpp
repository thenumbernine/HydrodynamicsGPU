#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <OpenCL/opencl.h>

#include "Common/Finally.h"
#include "Common/Exception.h"
#include <string>
#include <fstream>
#include <iostream>
#include <vector>
#include "Cell.h"

/*
kernels needed for burgers solver:
1) calc CFL	<- reduce
2) apply boundary	(or not since we will have already)
3) integrate flux
4) apply boundary
5) integrate pressure
6) apply boundary	(or not, since we will be again soon)

kernels needed for Roe solver:
1) calc CFL <- reduce
2) integrate flux
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
	  
	unsigned int correct;
 
	size_t global[DIM];
	size_t local[DIM] = {16, 16};
 
	cl_device_id deviceID;
	cl_context context;
	cl_kernel kernel;
	  
	std::string kernelSource = readFile("res/burgers.cl");

	real noise = real(.01);
	int i = 0;
	int size[DIM] = {1024, 1024};
	unsigned int count = size[0] * size[1];
	std::vector<Cell> data(count);
	{
		real xmin[DIM] = {.5, .5};
		real xmax[DIM] = {.5, .5};
		int index[DIM];
		
		int e = 0;
		Cell *cell = &data[0];
		//for (index[2] = 0; index[2] < size[2]; ++index[2]) {
			for (index[1] = 0; index[1] < size[1]; ++index[1]) {
				for (index[0] = 0; index[0] < size[0]; ++index[0], ++e, ++cell) {
					bool lhs = true;
					for (int n = 0; n < DIM; ++n) {
						cell->x[n] = real(xmax[n] - xmin[n]) * real(index[n]) / real(size[n]) + real(xmin[n]);
						if (cell->x[n] > real(.5) * real(xmax[n] + xmin[n])) {
							lhs = false;
						}
					}

					for (int m = 0; m < DIM; ++m) {
						for (int n = 0; n < DIM; ++n) {
							cell->interfaces[m].x[n] = cell->x[n];
							if (m == n) {
								cell->interfaces[m].x[n] -= real(xmax[n] - xmin[n]) * real(.5) / real(size[n]);
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

					cell->state[0] = density;
					for (int n = 0; n < DIM; ++n) {
						cell->state[n+1] = density * velocity[n];
					}
					cell->state[DIM+1] = density * energyTotal;

					cell->value = rand() / (real)RAND_MAX;
				}
			}
		//}
	}

	// Connect to a compute device
	//
	int gpu = 1;
	err = clGetDeviceIDs(NULL, gpu ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU, 1, &deviceID, NULL);
	if (err != CL_SUCCESS) {
		throw Exception() << "Error: Failed to create a device group!";
	}
  
	// Create a compute context 
	//
	context = clCreateContext(0, 1, &deviceID, NULL, NULL, &err);
	if (!context) {
		throw Exception() << "Error: Failed to create a compute context!";
	}
 
	// Create a command commands
	//
	cl_command_queue commands = clCreateCommandQueue(context, deviceID, 0, &err);
	if (!commands) {
		throw Exception() << "Error: Failed to create a command commands!";
	}
 
	// Create the compute program from the source buffer
	//
	const char *kernelSourcePtr = kernelSource.c_str();
	cl_program program = clCreateProgramWithSource(context, 1, (const char **) &kernelSourcePtr, NULL, &err);
	if (!program) {
		throw Exception() << "Error: Failed to create compute program!";
	}
 
	// Build the program executable
	//
	err = clBuildProgram(program, 0, NULL, "-I res/include", NULL, NULL);
	if (err != CL_SUCCESS) {
		size_t len;
		char buffer[2048];
 
		std::cout << "Error: Failed to build program executable!\n" << std::endl;
		clGetProgramBuildInfo(program, deviceID, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
		std::cout << buffer << std::endl;
		exit(1);
	}
 
	// Create the compute kernel in the program we wish to run
	//
	kernel = clCreateKernel(program, "square", &err);
	if (!kernel || err != CL_SUCCESS) {
		std::cout << "Error: Failed to create compute kernel!\n" << std::endl;
		exit(1);
	}
 
	// Create the input and output arrays in device memory for our calculation
	//
	cl_mem input = clCreateBuffer(context,  CL_MEM_READ_ONLY,  sizeof(Cell) * count, NULL, NULL);
	cl_mem output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(Cell) * count, NULL, NULL);
	if (!input || !output) {
		std::cout << "Error: Failed to allocate device memory!\n" << std::endl;
		exit(1);
	}	
	
	// Write our data set into the input array in device memory 
	//
	err = clEnqueueWriteBuffer(commands, input, CL_TRUE, 0, sizeof(Cell) * count, &data[0], 0, NULL, NULL);
	if (err != CL_SUCCESS) {
		std::cout << "Error: Failed to write to source array!\n" << std::endl;
		exit(1);
	}
 
	// Set the arguments to our compute kernel
	//
	err = 0;
	err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input);
	err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &output);
	err |= clSetKernelArg(kernel, 2, sizeof(cl_uint), &size[0]);
	err |= clSetKernelArg(kernel, 3, sizeof(cl_uint), &size[1]);
	if (err != CL_SUCCESS) {
		std::cout << "Error: Failed to set kernel arguments! " << err << std::endl;
		exit(1);
	}

/*
	why isn't this working?
	// Get the maximum work group size for executing the kernel on the device
	//
	err = clGetKernelWorkGroupInfo(kernel, deviceID, CL_KERNEL_WORK_GROUP_SIZE, 2 * sizeof(local), local, NULL);
	if (err != CL_SUCCESS) {
		std::cout << "Error: Failed to retrieve kernel work group info! " << err << std::endl;
		exit(1);
	}
*/ 
	// Execute the kernel over the entire range of our 1d input data set
	// using the maximum number of work group items for this device
	//
	for (int n = 0; n < DIM; ++n) {
		global[n] = size[n];
	}
	err = clEnqueueNDRangeKernel(commands, kernel, 2, NULL, global, local, 0, NULL, NULL);
	if (err) {
		throw Exception() << "Error: Failed to execute kernel!";
	}
 
	// Wait for the command commands to get serviced before reading back results
	//
	clFinish(commands);
 
	// Read back the results from the device to verify the output
	//
	std::vector<Cell> results(count);
	err = clEnqueueReadBuffer( commands, output, CL_TRUE, 0, sizeof(Cell) * count, &results[0], 0, NULL, NULL );  
	if (err != CL_SUCCESS) {
		std::cout << "Error: Failed to read output array! " << err << std::endl;
		exit(1);
	}
	
	// Validate our results
	//
	correct = 0;
	for(i = 0; i < count; i++) {
		if(results[i].value == data[i].value * data[i].value)
			correct++;
	}
	
	// Print a brief summary detailing the results
	//
	std::cout << "Computed '" << correct << "/" << count << "' correct values!" << std::endl;
	
	// Shutdown and cleanup
	//
	clReleaseMemObject(input);
	clReleaseMemObject(output);
	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseCommandQueue(commands);
	clReleaseContext(context);
 
	return 0;
}
