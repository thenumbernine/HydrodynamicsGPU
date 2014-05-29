#include "HydroGPU/RoeSolver.h"
#include "Common/Exception.h"
#include "Common/Finally.h"
#include "Common/Macros.h"
#include "TensorMath/Vector.h"
#include <OpenGL/gl.h>
#include <fstream>

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

RoeSolver::RoeSolver(
	cl_device_id deviceID,
	cl_context context,
	cl_int2 size_,
	cl_command_queue commands,
	std::vector<Cell> &cells,
	real* xmin,
	real* xmax,
	cl_mem fluidTexMem,
	cl_mem gradientTexMem,
	size_t *local_size,
	bool useGPU_)
: Solver(deviceID, context, size_, commands, cells, xmin, xmax, fluidTexMem, gradientTexMem, local_size, useGPU_)
, program(cl_program())
, cellsMem(cl_mem())
, cflMem(cl_mem())
, cflTimestepMem(cl_mem())
, calcEigenDecompositionKernel(cl_kernel())
, calcCFLAndDeltaQTildeKernel(cl_kernel())
, calcCFLMinReduceKernel(cl_kernel())
, calcCFLMinFinalKernel(cl_kernel())
, calcRTildeKernel(cl_kernel())
, calcFluxKernel(cl_kernel())
, updateStateKernel(cl_kernel())
, convertToTexKernel(cl_kernel())
, useGPU(useGPU_)
, cfl(.5f)
{
	int err = 0;
	size = size_;

	std::string kernelSource = readFile("res/roe_euler_2d.cl");
	const char *kernelSourcePtr = kernelSource.c_str();
	cl_program program = clCreateProgramWithSource(context, 1, (const char **) &kernelSourcePtr, NULL, &err);
	if (!program) throw Exception() << "Error: Failed to create compute program!";
 
	err = clBuildProgram(program, 0, NULL, "-I res/include", NULL, NULL);
	if (err != CL_SUCCESS) {
 
		std::cout << "failed to build program executable!" << std::endl;
		
		size_t len;
		clGetProgramBuildInfo(program, deviceID, CL_PROGRAM_BUILD_LOG, 0, NULL, &len);
		std::string log(len, '\0');
		clGetProgramBuildInfo(program, deviceID, CL_PROGRAM_BUILD_LOG, len, (void*)log.c_str(), NULL);
		std::cout << log << std::endl;
		exit(1);
	}
 
	unsigned int count = size.s[0] * size.s[1];
	cellsMem = clCreateBuffer(context,  CL_MEM_READ_WRITE,  sizeof(Cell) * count, NULL, NULL);
	if (!cellsMem) throw Exception() << "failed to allocate device memory!";

	cflMem = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(real) * count, NULL, NULL);
	if (!cflMem) throw Exception() << "failed to allocate device memory";
	
	cflTimestepMem = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(real), NULL, NULL);
	if (!cflTimestepMem) throw Exception() << "failed to allocate device memory";
	
	err = clEnqueueWriteBuffer(commands, cellsMem, CL_TRUE, 0, sizeof(Cell) * count, &cells[0], 0, NULL, NULL);
	if (err != CL_SUCCESS) {
		std::cout << "Error: Failed to write to source array!\n" << std::endl;
		exit(1);
	}
 
	calcEigenDecompositionKernel = clCreateKernel(program, "calcEigenDecomposition", &err);
	if (!calcEigenDecompositionKernel || err != CL_SUCCESS) throw Exception() << "failed to create kernel";

	calcCFLAndDeltaQTildeKernel = clCreateKernel(program, "calcCFLAndDeltaQTilde", &err);
	if (!calcCFLAndDeltaQTildeKernel || err != CL_SUCCESS) throw Exception() << "failed to create kernel with error " << err;
	
	calcCFLMinReduceKernel = clCreateKernel(program, "calcCFLMinReduce", &err);
	if (!calcCFLMinReduceKernel || err != CL_SUCCESS) throw Exception() << "failed to create kernel with error " << err;

	calcCFLMinFinalKernel = clCreateKernel(program, "calcCFLMinFinal", &err);
	if (!calcCFLMinFinalKernel || err != CL_SUCCESS) throw Exception() << "failed to create kernel with error " << err;

	calcRTildeKernel = clCreateKernel(program, "calcRTilde", &err);
	if (!calcRTildeKernel || err != CL_SUCCESS) throw Exception() << "failed to create kernel";

	calcFluxKernel = clCreateKernel(program, "calcFlux", &err);
	if (!calcFluxKernel || err != CL_SUCCESS) throw Exception() << "failed to create kernel";
	
	updateStateKernel = clCreateKernel(program, "updateState", &err);
	if (!updateStateKernel || err != CL_SUCCESS) throw Exception() << "failed to create kernel";

	//if (useGPU) 
	{
		convertToTexKernel = clCreateKernel(program, "convertToTex", &err);
		if (!convertToTexKernel || err != CL_SUCCESS) throw Exception() << "failed to create kernel";
	}

	cl_kernel* kernels[] = {
		&calcEigenDecompositionKernel,
		&calcCFLAndDeltaQTildeKernel,
		&calcRTildeKernel,
		&calcFluxKernel,
		&updateStateKernel,
	};
	std::for_each(kernels, kernels + numberof(kernels), [&](cl_kernel* kernel) {
		err = 0;
		err  = clSetKernelArg(*kernel, 0, sizeof(cl_mem), &cellsMem);
		err |= clSetKernelArg(*kernel, 1, sizeof(cl_uint2), &size.s[0]);
		if (err != CL_SUCCESS) throw Exception() << "Error: Failed to set kernel arguments! " << err;
	});

	real2 dx;
	for (int i = 0; i < DIM; ++i) {
		dx.s[i] = (xmax[i] - xmin[i]) / (float)size.s[i];
	}
	
	err = clSetKernelArg(calcFluxKernel, 2, sizeof(real2), dx.s);
	err |= clSetKernelArg(calcFluxKernel, 3, sizeof(cl_mem), &cflTimestepMem);
	if (err != CL_SUCCESS) throw Exception() << "Error: Failed to set kernel arguments! " << err;

	err = clSetKernelArg(updateStateKernel, 2, sizeof(real2), dx.s);
	err |= clSetKernelArg(updateStateKernel, 3, sizeof(cl_mem), &cflTimestepMem);
	if (err != CL_SUCCESS) throw Exception() << "Error: Failed to set kernel arguments! " << err;
	
	err = clSetKernelArg(calcCFLAndDeltaQTildeKernel, 2, sizeof(cl_mem), &cflMem);
	err |= clSetKernelArg(calcCFLAndDeltaQTildeKernel, 3, sizeof(real2), dx.s);
	if (err != CL_SUCCESS) throw Exception() << "failed to set argument with error " << err;

	err = clSetKernelArg(calcCFLMinReduceKernel, 0, sizeof(cl_mem), &cflMem);
	err |= clSetKernelArg(calcCFLMinReduceKernel, 1, local_size[0] * sizeof(real), NULL);
	if (err != CL_SUCCESS) throw Exception() << "failed to set argument with error " << err;
	
	err = clSetKernelArg(calcCFLMinFinalKernel, 0, sizeof(cl_mem), &cflMem);
	err |= clSetKernelArg(calcCFLMinFinalKernel, 1, local_size[0] * sizeof(real), NULL);
	err |= clSetKernelArg(calcCFLMinFinalKernel, 2, sizeof(cl_mem), &cflTimestepMem);
	err |= clSetKernelArg(calcCFLMinFinalKernel, 3, sizeof(real), &cfl);
	if (err != CL_SUCCESS) throw Exception() << "failed to set argument with error " << err;

	//if (useGPU) 
	{
		err = 0;
		err  = clSetKernelArg(convertToTexKernel, 0, sizeof(cl_mem), &cellsMem);
		err |= clSetKernelArg(convertToTexKernel, 1, sizeof(cl_uint2), &size.s[0]);
		err |= clSetKernelArg(convertToTexKernel, 2, sizeof(cl_mem), &fluidTexMem);
		err |= clSetKernelArg(convertToTexKernel, 3, sizeof(cl_mem), &gradientTexMem);
		if (err != CL_SUCCESS) throw Exception() << "Error: Failed to set kernel arguments! " << err;
	}
}

RoeSolver::~RoeSolver() {
	clReleaseProgram(program);
	clReleaseMemObject(cellsMem);
	clReleaseMemObject(cflMem);
	clReleaseMemObject(cflTimestepMem);
	clReleaseKernel(calcEigenDecompositionKernel);
	clReleaseKernel(calcCFLAndDeltaQTildeKernel);
	clReleaseKernel(calcCFLMinReduceKernel);
	clReleaseKernel(calcCFLMinFinalKernel);
	clReleaseKernel(calcRTildeKernel);
	clReleaseKernel(calcFluxKernel);
	clReleaseKernel(updateStateKernel);
	if (convertToTexKernel) clReleaseKernel(convertToTexKernel);
}

void RoeSolver::update(
	cl_command_queue commands, 
	cl_mem fluidTexMem, 
	size_t *global_size,
	size_t *local_size)
{
	int err = 0;

	err = clEnqueueNDRangeKernel(commands, calcEigenDecompositionKernel, 2, NULL, global_size, local_size, 0, NULL, NULL);
	if (err) throw Exception() << "failed to execute calcEigenDecompositionKernel with error " << err;
	
	err = clEnqueueNDRangeKernel(commands, calcCFLAndDeltaQTildeKernel, 2, NULL, global_size, local_size, 0, NULL, NULL);
	if (err) throw Exception() << "failed to execute calcCFLAndDeltaQTildeKernel with error " << err;

	size_t reduceGlobalSize = global_size[0] * global_size[1] / 4;
	err = clEnqueueNDRangeKernel(commands, calcCFLMinReduceKernel, 1, NULL, &reduceGlobalSize, local_size, 0, NULL, NULL);
	if (err) throw Exception() << "failed to execute calcCFLMinReduceKernel with error " << err;
	
	while (reduceGlobalSize / local_size[0] > local_size[0]) {
		reduceGlobalSize /= local_size[0];
		err = clEnqueueNDRangeKernel(commands, calcCFLMinReduceKernel, 1, NULL, &reduceGlobalSize, local_size, 0, NULL, NULL);
		if (err) throw Exception() << "failed to execute calcCFLMinReduceKernel loop with error " << err;
	}
	reduceGlobalSize /= local_size[0];

	err = clSetKernelArg(calcCFLMinFinalKernel, 4, sizeof(size_t), &reduceGlobalSize);
	if (err != CL_SUCCESS) throw Exception() << "failed to set kernel arguments! " << err;

	err = clEnqueueNDRangeKernel(commands, calcCFLMinFinalKernel, 1, NULL, local_size, local_size, 0, NULL, NULL);
	if (err) throw Exception() << "failed to execute calcCFLMinFinalKernel with error " << err;
	
	err = clEnqueueNDRangeKernel(commands, calcRTildeKernel, 2, NULL, global_size, local_size, 0, NULL, NULL);
	if (err) throw Exception() << "failed to execute calcRTildeKernel with error " << err;
	
	err = clEnqueueNDRangeKernel(commands, calcFluxKernel, 2, NULL, global_size, local_size, 0, NULL, NULL);
	if (err) throw Exception() << "failed to execute calcFluxKernel with error " << err;
	
	err = clEnqueueNDRangeKernel(commands, updateStateKernel, 2, NULL, global_size, local_size, 0, NULL, NULL);
	if (err) throw Exception() << "failed to execute updateStateKernel with error " << err;

	glFlush();
	glFinish();
	clEnqueueAcquireGLObjects(commands, 1, &fluidTexMem, 0, 0, 0);

	if (useGPU) {
		err = clEnqueueNDRangeKernel(commands, convertToTexKernel, 2, NULL, global_size, local_size, 0, NULL, NULL);
		if (err) throw Exception() << "failed to execute convertToTexKernel";
	} else {
		int count = size.s[0] * size.s[1];
		std::vector<Cell> cells(count);
		err = clEnqueueReadBuffer(commands, cellsMem, CL_TRUE, 0, sizeof(Cell) * count, &cells[0], 0, NULL, NULL);  
		if (err != CL_SUCCESS) throw Exception() << "Error: Failed to read cellsMem array! " << err;
		std::vector<Vector<char,4>> buffer(count);
		for (int i = 0; i < count; ++i) {
			buffer[i](0) = (char)(255.f * cells[i].q.s[0] * .9f);
		}
		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, size.s[0], size.s[1], GL_RGBA, GL_UNSIGNED_BYTE, &buffer[0].v);
	}

	clEnqueueReleaseGLObjects(commands, 1, &fluidTexMem, 0, 0, 0);
	clFlush(commands);
	clFinish(commands);
}

void RoeSolver::addDrop(float x, float y, float dx, float dy) {
}


